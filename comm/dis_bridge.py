"""Connection of AircraftEnv to VR-Forces over DIS, created by AircraftEnv for config["dis"] (see test_dis.py).

The aircraft with mission model_vrf are simulated by VR-Forces: each is bound to a VR-F entity by its marking, and its JSBSim is
replaced by DISRemoteFDM, which dead reckons the entity to every JSBSim step. All other aircraft are simulated by the JSBSim of
the env and sent to VR-F. The env runs in real time. Hits are handled like the ones of the cannon of the env (CannonSim):
- A detonation of VR-F closer than kill_radius to an aircraft of the env hits it with the hit probability of the cannon.
- A hit of the cannon of the env on a VR-F aircraft is sent to VR-F as detonation PDU (after the fire PDU of the burst),
  so that VR-F destroys it.
"""
import math
import random
import time

import numpy as np

from comm.dis_comm import (DAMAGE_DESTROYED, DEACTIVATED, FLAMING, NO_ENTITY, POWER_PLANT_ON, SMOKE_PLUME, DISComm,
                           detonation_result_name, dis_to_geodetic, geodetic_to_dis, munition_name)
from env.sim import CannonSim

FEET_TO_METERS = 0.3048

# DIS force per team: friendly, opposing. Other teams are neutral.
FORCE_IDS = {"B": 1, "R": 2}
FORCE_NEUTRAL = 3

# DIS entity types per JSBSim model of the aircraft sent to VR-F
ENTITY_TYPES = {"f16": [1, 2, 225, 1, 2, 1, 0], "f15": [1, 2, 225, 1, 5, 1, 0]}

# Munition of the fire and detonation PDUs of the cannon of the env: AIM-120C (munition, anti-air, USA, guided, AMRAAM, C),
# with high explosive blast fragmentation warhead and proximity fuse. VR-F's damage tables decide whether it destroys the target,
# a single detonation of a gun round would hardly destroy an aircraft.
MUNITION = ([2, 1, 225, 1, 2, 3, 0], 1620, 3000)


class DISBridge:
    """Connects an AircraftEnv to VR-Forces over DIS, see the module docstring.

    Args:
        env: the AircraftEnv.
        address, port, exercise_id, site_id, application_id, receive_port: the DIS connection, see comm.dis_comm.DISComm.
        vrf_entities: {aid: marking} for the aircraft with mission model_vrf: the marking of their VR-F entity, e.g. "F-35A 1",
            or None for any air platform of the force of their team (B friendly, R opposing) received over DIS.
        entity_types: DIS entity types of the aircraft sent to VR-F per JSBSim model, by default ENTITY_TYPES. Other models are
            sent with the first type.
        munition: (entity type, warhead, fuse) of the fire and detonation PDUs of the cannon of the env, by default MUNITION.
        send_rate: entity state PDUs per second of each aircraft sent to VR-F, shot down aircraft are sent every 2 seconds.
        entity_timeout: seconds without entity state PDU after which a VR-F aircraft counts as removed (crashed).
        kill_radius: m, VR-F detonations closer to an aircraft of the env hit it with the hit probability of the cannon of the env.
        wait_timeout: seconds to wait for the VR-F entities at the start of an episode, None to wait as long as needed.
        verbose: print the bindings of the VR-F aircraft, the hits and the problems of the connection.
    """

    MAX_LAG = 0.5             # s behind real time, after which the simulation continues from now instead of catching up
    DESTROYED_HEARTBEAT = 2.0  # s between the entity states of shot down aircraft
    REPORT_INTERVAL = 5.0      # s between the reports of connection problems and of the waiting for VR-F entities

    def __init__(self, env, address="10.11.255.255", port=3000, exercise_id=1, site_id=1, application_id=8492,
                 vrf_entities=None, entity_types=None, munition=None, send_rate=25.0, entity_timeout=12.0, kill_radius=50.0,
                 wait_timeout=None, receive_port=None, verbose=True):
        self.selectors = dict(vrf_entities or {})
        vrf_aids = [aid for aid, config in env.config.aircraft.items() if config.get("mission") == "model_vrf"]
        if sorted(self.selectors) != sorted(vrf_aids):
            raise ValueError(f"vrf_entities {sorted(self.selectors)} must name each aircraft with mission model_vrf {vrf_aids}")
        self.vrf_aids = vrf_aids
        self.local_aids = [aid for aid in env.agents if aid not in self.selectors]
        self.entity_numbers = {aid: i + 1 for i, aid in enumerate(self.local_aids)}
        entity_types = entity_types or ENTITY_TYPES
        default_type = next(iter(entity_types.values()))
        self.entity_types = {aid: entity_types.get(env.agents[aid].model, default_type) for aid in self.local_aids}
        self.force_ids = {aid: FORCE_IDS.get(aid[0], FORCE_NEUTRAL) for aid in env.agents}
        self.markings = {aid: f"{env.agents[aid].model.upper()} {'RL' if env.agents[aid].is_learning else 'SIM'}-{aid}"
                         for aid in self.local_aids}  # e.g. "F16 RL-B1", shown by VR-F
        self.munition = munition or MUNITION
        self.dt = 1.0 / env.sim_freq
        self.send_interval = max(1, round(env.sim_freq / send_rate))
        self.heartbeat_interval = max(1, round(env.sim_freq * self.DESTROYED_HEARTBEAT))
        self.entity_timeout = entity_timeout
        self.kill_radius = kill_radius
        self.wait_timeout = wait_timeout
        self.verbose = verbose

        self.dis = DISComm(address, port, exercise_id, site_id, application_id, receive_port)
        self.bound = {aid: None for aid in self.vrf_aids}  # ID of the VR-F entity of each VR-F aircraft
        for aid in self.vrf_aids:
            env.agents[aid].fdm_factory = self.create_fdm

        self._wall_start = time.perf_counter()
        self._sim_time = 0.0
        self._substeps = 0
        self._shots = {}          # remaining cannon shots per aircraft of the env, to find its bursts
        self._fire_events = {}    # event number of the last burst per aircraft of the env
        self._sent_hits = set()   # VR-F aircraft hit by the cannon of the env, sent to VR-F
        self._destroyed_in_vrf = set()
        self._sent_alive = {}     # alive status of the last entity state sent per aircraft of the env
        self._last_states = {}    # (location, orientation) of the last entity state sent per aircraft of the env
        self._last_report = 0.0
        self._print(f"DIS exercise {exercise_id} at {address}:{port}: sending {self.local_aids} as site {site_id}, "
                    f"application {application_id}" + (f", receiving {self.vrf_aids} from VR-F" if self.vrf_aids else ""))

    @property
    def time(self):
        """Simulated seconds since the start of the episode, which run in real time."""
        return self._sim_time

    def create_fdm(self, agent):
        """AircraftSim.fdm_factory of the VR-F aircraft."""
        return DISRemoteFDM(self, agent.aid)

    def entity(self, aid):
        """The RemoteEntity bound to the VR-F aircraft aid."""
        return self.dis.entities[self.bound[aid]]

    def world_time(self):
        """time.perf_counter() of the end of the current JSBSim step, to which the VR-F aircraft are dead reckoned."""
        return self._wall_start + self._sim_time + self.dt

    def label(self, aid):
        """aid, with the marking of its VR-F entity for VR-F aircraft."""
        key = self.bound.get(aid)
        return f"{aid} ({self.dis.entities[key].marking})" if key is not None else aid

    def traffic(self):
        """Summary of the DIS traffic."""
        stats = self.dis.stats
        return (f"DIS: {stats['received']} received ({stats['entity_states']} entity states, {stats['detonations']} "
                f"detonations, {stats['other']} other packets), {stats['sent']} sent")

    def before_reset(self, env):
        """Called at the start of AircraftEnv.reset(): binds the VR-F aircraft to VR-F entities, waiting until all are received."""
        start = time.perf_counter()
        last_report = start - self.REPORT_INTERVAL + 1.0  # the first report after one second of waiting
        while True:
            self.dis.poll()
            now = time.perf_counter()
            unbound = self._bind(now)
            if not unbound:
                break
            if self.wait_timeout is not None and now - start > self.wait_timeout:
                raise TimeoutError(f"No VR-F entities for {self._describe(unbound)} within {self.wait_timeout} s. "
                                   f"Received entities:\n{self.dis.entity_table(now)}")
            if now - last_report >= self.REPORT_INTERVAL:
                last_report = now
                print(f"Waiting for the VR-F entities of {self._describe(unbound)} (DIS exercise {self.dis.exercise_id}, port "
                      f"{self.dis.port}). Received entities:\n{self.dis.entity_table(now)}")
            time.sleep(0.05)
        # The VR-F aircraft are dead reckoned to this time when AircraftSim.reload() creates their DISRemoteFDM
        self._wall_start, self._sim_time = time.perf_counter(), 0.0

    def on_reset(self, env):
        """Called at the end of AircraftEnv.reset(): restarts the real-time clock and sends the aircraft of the env at their
        initial states."""
        self._wall_start, self._sim_time, self._substeps = time.perf_counter(), 0.0, 0
        self._shots = {aid: env.agents[aid].cannon.cannon_shots for aid in self.local_aids}
        self._fire_events.clear()
        self._sent_hits.clear()
        self._destroyed_in_vrf.clear()
        self.dis.detonations.clear()  # of the last episode
        for aid in self.vrf_aids:
            env.agents[aid].ac_sim.update()
            env.agents[aid].update_properties()
        self._send_states(env, self.local_aids)

    def before_substep(self, env):
        """Called by AircraftEnv.run_substep() before the aircraft are advanced by one JSBSim step: receives the PDUs of VR-F
        and applies its destroyed and removed entities and its detonations."""
        self.dis.poll()
        now = time.perf_counter()
        for aid in self.vrf_aids:
            agent, entity = env.agents[aid], self.entity(aid)
            if not agent.alive:
                continue
            if entity.destroyed:
                agent.shootdown()
                self._destroyed_in_vrf.add(aid)
                self._print(f"VR-F aircraft {self.label(aid)} was destroyed in VR-F")
            elif entity.deactivated or entity.age(now) > self.entity_timeout:
                agent.crash()
                self._print(f"VR-F aircraft {self.label(aid)} was removed from VR-F"
                            + ("" if entity.deactivated else f" (no entity state for {entity.age(now):.1f} s)"))
        while self.dis.detonations:
            self._apply_detonation(env, self.dis.detonations.popleft())
        if now - self._last_report >= self.REPORT_INTERVAL:
            error = self.dis.pop_error()
            if error:
                self._last_report = now
                self._print(f"DIS {error} ({self.traffic()})")

    def after_substep(self, env):
        """Called by AircraftEnv.run_substep() after the aircraft were advanced by one JSBSim step: sends the bursts and hits of
        the cannon of the env and the states of its aircraft, and keeps real time."""
        self._sim_time += self.dt
        self._substeps += 1
        self._send_cannon_events(env)
        due = self._substeps % self.send_interval == 0
        heartbeat = self._substeps % self.heartbeat_interval == 0
        aids = [aid for aid in self.local_aids
                if env.agents[aid].alive != self._sent_alive.get(aid)  # shot down or crashed: show it at once
                or (due if env.agents[aid].alive else heartbeat)]
        if aids:
            self._send_states(env, aids)
        self._keep_real_time()

    def close(self):
        """Removes the aircraft of the env from VR-F (deactivated entity state) and closes the connection."""
        if self.dis is None:
            return
        for _ in range(3):  # UDP may lose a datagram
            for aid, (location, orientation) in self._last_states.items():
                self.dis.send_entity_state(self.entity_numbers[aid], self.force_ids[aid], self.entity_types[aid],
                                           self.markings[aid], location, np.zeros(3), orientation, np.zeros(3), DEACTIVATED)
        self.dis.close()
        self.dis = None

    def _bind(self, now):
        """Binds the unbound VR-F aircraft to active VR-F entities. @returns the VR-F aircraft that remain unbound."""
        for aid, key in self.bound.items():
            if key is not None and not self._active(self.dis.entities[key], now):
                self.bound[aid] = None
        taken = {key for key in self.bound.values() if key is not None}
        unbound = []
        for aid, selector in self.selectors.items():
            if self.bound[aid] is not None:
                continue
            candidates = [entity for entity in self.dis.entities.values()
                          if entity.id not in taken and self._active(entity, now) and self._matches(entity, aid, selector)]
            if not candidates:
                unbound.append(aid)
                continue
            # A marking: the entity updated last, e.g. after a restart of the scenario. Any entity: the one received first.
            entity = (max(candidates, key=lambda e: e.received) if selector is not None
                      else min(candidates, key=lambda e: e.first_seen))
            self.bound[aid] = entity.id
            taken.add(entity.id)
            self._print(f"{aid} is the VR-F entity '{entity.marking}' ({':'.join(map(str, entity.id))})")
        return unbound

    def _active(self, entity, now):
        return not entity.destroyed and not entity.deactivated and entity.age(now) <= self.entity_timeout

    @staticmethod
    def _matches(entity, aid, selector):
        if selector is not None:
            return entity.marking == selector
        return entity.is_air_platform and entity.force_id == FORCE_IDS.get(aid[0], FORCE_NEUTRAL)

    def _describe(self, aids):
        return ", ".join(f"{aid} ('{self.selectors[aid]}')" if self.selectors[aid] is not None
                         else f"{aid} (any air platform of force {FORCE_IDS.get(aid[0], FORCE_NEUTRAL)})" for aid in aids)

    def _dis_id(self, aid):
        """DIS entity ID of an aircraft of the env or of the VR-F entity of a VR-F aircraft."""
        if aid in self.entity_numbers:
            return (*self.dis.simulation_address, self.entity_numbers[aid])
        return self.bound.get(aid) or NO_ENTITY

    def _local_state(self, agent):
        """ECEF location, ECEF velocity, DIS Euler angles and angular velocity (body axes) of an aircraft of the env."""
        get = agent.ac_sim.get_property_value
        velocity_ned = np.array([get("velocities/v-north-fps"), get("velocities/v-east-fps"),
                                 get("velocities/v-down-fps")]) * FEET_TO_METERS
        location, velocity, orientation = geodetic_to_dis(
            get("position/lat-geod-rad"), get("position/long-gc-rad"), get("position/h-sl-meters"),
            get("attitude/roll-rad"), get("attitude/pitch-rad"), get("attitude/heading-true-rad"), velocity_ned)
        rates = np.array([get("velocities/p-rad_sec"), get("velocities/q-rad_sec"), get("velocities/r-rad_sec")])
        return location, velocity, orientation, rates

    def _send_states(self, env, aids):
        for aid in aids:
            agent = env.agents[aid]
            location, velocity, orientation, rates = self._local_state(agent)
            if agent.alive:
                appearance = POWER_PLANT_ON
            else:
                appearance = DAMAGE_DESTROYED | SMOKE_PLUME | FLAMING
                velocity, rates = np.zeros(3), np.zeros(3)  # no longer simulated
            self.dis.send_entity_state(self.entity_numbers[aid], self.force_ids[aid], self.entity_types[aid],
                                       self.markings[aid], location, velocity, orientation, rates, appearance)
            self._sent_alive[aid] = agent.alive
            self._last_states[aid] = (location, orientation)

    def _apply_detonation(self, env, detonation):
        """Hits the aircraft of the env closest to a VR-F detonation, if within kill_radius, with the hit probability of the
        cannon of the env. Like the cannon of the env, a VR-F teammate does not hit it. Each detonation is printed."""
        shooter = next((aid for aid, key in self.bound.items() if key == detonation.firing_id), None)
        description = (f"VR-F detonation of {self._shooter_name(shooter, detonation.firing_id)} "
                       f"[munition {munition_name(detonation.munition_type)}, {detonation_result_name(detonation.result)}]")
        distances = [(float(np.linalg.norm(self._local_state(env.agents[aid])[0] - detonation.location)), aid)
                     for aid in self.local_aids
                     if env.agents[aid].alive and not (shooter is not None and shooter[0] == aid[0])]
        if not distances:
            self._print(f"{description}, no aircraft of the env it can hit")
            return
        distance, aid = min(distances)
        if distance > self.kill_radius:
            target = "its target " if detonation.target_id == self._dis_id(aid) else ""
            self._print(f"{description} {distance:.0f} m from {target}{aid}, outside the kill radius ({self.kill_radius:.0f} m)")
            return
        hit = random.random() < CannonSim.HIT_PROBABILITY
        if hit:
            env.agents[aid].shootdown()
        self._print(f"{description} {distance:.0f} m from {aid}: {'hit' if hit else 'missed'} "
                    f"({CannonSim.HIT_PROBABILITY:.0%} hit probability)")

    def _shooter_name(self, shooter, firing_id):
        """Name of the firing entity of a detonation: the VR-F aircraft of the env with its marking, else the marking of the
        received entity, else its DIS ID."""
        if shooter is not None:
            return self.label(shooter)
        entity = self.dis.entities.get(firing_id)
        dis_id = ":".join(map(str, firing_id))
        return f"'{entity.marking}' ({dis_id})" if entity is not None and entity.marking else dis_id

    def _send_cannon_events(self, env):
        """Sends a fire PDU for each burst of the cannon of an aircraft of the env (CannonSim.launch()), and a detonation PDU
        for each hit of a VR-F aircraft, so that VR-F destroys it."""
        for aid in self.local_aids:
            agent = env.agents[aid]
            if agent.cannon.cannon_shots >= self._shots[aid]:
                continue
            self._shots[aid] = agent.cannon.cannon_shots
            location, velocity, _, _ = self._local_state(agent)
            closest = agent.mission.get_closest_opp(agent)
            target_id, distance = (self._dis_id(closest[0][0].aid), closest[0][1]) if closest else (NO_ENTITY, 0.0)
            self._fire_events[aid] = self.dis.new_event_number()
            self.dis.send_fire(self.entity_numbers[aid], target_id, self._fire_events[aid], location, velocity,
                               self.munition, distance)
        for aid in self.vrf_aids:
            agent = env.agents[aid]
            if not agent.shotdown or aid in self._sent_hits or aid in self._destroyed_in_vrf:
                continue
            self._sent_hits.add(aid)
            shooter = self._shooter(env, agent)
            entity = self.entity(aid)
            location, velocity, _ = entity.state_at(self._wall_start + self._sim_time)
            event_number = self._fire_events.get(shooter) or self.dis.new_event_number()
            self.dis.send_detonation(self.entity_numbers.get(shooter, 0), entity.id, event_number, location, velocity,
                                     self.munition)
            self._print(f"{shooter or 'The cannon'} hit the VR-F aircraft {self.label(aid)}, detonation sent to VR-F")

    def _shooter(self, env, target):
        """The aircraft of the env whose cannon hit the target in this JSBSim step (CannonSim.success until the mission
        reverts it at the end of the env step), the closest one if several hit."""
        shooters = [aid for aid in self.local_aids
                    if env.agents[aid].cannon.success and target in env.agents[aid].opponents]
        return min(shooters, key=lambda aid: env.agents[aid].get_distance_to_target(target), default=None)

    def _keep_real_time(self):
        """Waits until the real time of the end of the JSBSim step. Far behind real time, e.g. after the program was paused,
        the simulation continues from now instead of catching up."""
        target = self._wall_start + self._sim_time
        now = time.perf_counter()
        if now < target:
            time.sleep(target - now)
        elif now - target > self.MAX_LAG:
            self._wall_start += now - target
            self._print(f"The simulation was {now - target:.2f} s behind real time, it continues from now")

    def _print(self, message):
        if self.verbose:
            print(message)


class DISRemoteFDM:
    """Replaces the jsbsim.FGFDMExec of a VR-F aircraft (mission model_vrf), like rendering.flightsim.FlightSimFDM.

    run() dead reckons the VR-F entity bound to the aircraft to the end of the JSBSim step and computes the JSBSim properties of
    its state. The properties set by the env (flight controls, initial conditions) are stored, but have no effect on VR-F.
    """

    def __init__(self, bridge, aid):
        self.bridge = bridge
        self.aid = aid
        self.properties = {}
        self.start_time = None

    def set_debug_level(self, level):
        pass

    def set_aircraft_path(self, path):
        pass

    def load_model(self, model):
        return True

    def set_dt(self, dt):
        pass

    def query_property_catalog(self, check):
        return ""  # no JSBSim properties to add to the Catalog

    def get_property_value(self, name):
        try:
            return self.properties[name]
        except KeyError:
            raise KeyError(f"The property '{name}' of the VR-F aircraft {self.aid} is not known from DIS") from None

    def set_property_value(self, name, value):
        self.properties[name] = value

    def run_ic(self):
        self.update()
        return True

    def run(self):
        self.update()
        return True

    def get_sim_time(self):
        return self.properties["simulation/sim-time-sec"]

    def get_propulsion(self):
        return _NoPropulsion()

    def update(self):
        """Computes the properties of the state of the VR-F entity at the end of the current JSBSim step."""
        now = self.bridge.world_time()
        if self.start_time is None:
            self.start_time = now
        entity = self.bridge.entity(self.aid)
        location, velocity, orientation = entity.state_at(now)
        (lat, lon, alt), (roll, pitch, yaw), velocity_ned = dis_to_geodetic(location, velocity, orientation)
        yaw %= 2 * math.pi  # [0, 2 pi) as in JSBSim
        u, v, w = orientation @ velocity  # body axes, without wind
        speed = float(np.linalg.norm(velocity))
        p, q, r = entity.angular_velocity
        north, east, down = velocity_ned / FEET_TO_METERS
        self.properties.update({
            "position/lat-geod-rad": lat,
            "position/lat-geod-deg": math.degrees(lat),
            "position/lat-gc-rad": math.atan2(location[2], math.hypot(location[0], location[1])),
            "position/long-gc-rad": lon,
            "position/long-gc-deg": math.degrees(lon),
            "position/h-sl-meters": alt,
            "position/h-sl-ft": alt / FEET_TO_METERS,
            "attitude/phi-rad": roll,
            "attitude/roll-rad": roll,
            "attitude/phi-deg": math.degrees(roll),
            "attitude/theta-rad": pitch,
            "attitude/pitch-rad": pitch,
            "attitude/theta-deg": math.degrees(pitch),
            "attitude/psi-rad": yaw,
            "attitude/heading-true-rad": yaw,
            "attitude/psi-deg": math.degrees(yaw),
            "velocities/v-north-fps": north,
            "velocities/v-east-fps": east,
            "velocities/v-down-fps": down,
            "velocities/u-fps": u / FEET_TO_METERS,
            "velocities/v-fps": v / FEET_TO_METERS,
            "velocities/w-fps": w / FEET_TO_METERS,
            "velocities/vt-fps": speed / FEET_TO_METERS,
            "velocities/vg-fps": math.hypot(north, east),
            "velocities/vc-fps": calibrated_airspeed(speed, alt) / FEET_TO_METERS,
            "velocities/p-rad_sec": p,
            "velocities/q-rad_sec": q,
            "velocities/r-rad_sec": r,
            "aero/alpha-rad": math.atan2(w, u),
            "aero/beta-rad": math.asin(max(-1.0, min(1.0, v / speed))) if speed > 0 else 0.0,
            "simulation/sim-time-sec": now - self.start_time,
        })


class _NoPropulsion:
    """The part of jsbsim.FGPropulsion used by AircraftSim, for a VR-F aircraft, whose engines are simulated by VR-F."""

    def get_num_engines(self):
        return 0

    def get_steady_state(self):
        return True


def calibrated_airspeed(true_airspeed, altitude):
    """Approximate calibrated airspeed (m/s) of a true airspeed (m/s) at an altitude (m), as equivalent airspeed in the
    standard atmosphere (troposphere, without compressibility correction and wind)."""
    density_ratio = (1.0 - 2.25577e-5 * min(max(altitude, 0.0), 11000.0)) ** 4.2559
    return true_airspeed * math.sqrt(density_ratio)
