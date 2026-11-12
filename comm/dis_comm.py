"""DIS (IEEE 1278.1) communication over UDP with opendis (https://github.com/open-dis/open-dis-python), e.g. with VR-Forces.

DISComm sends entity state, fire and detonation PDUs, and receives the entity states (RemoteEntity) and detonations
(Detonation) of the other applications of the exercise. It is polled from the simulation loop instead of a receiver thread,
the operating system buffers the datagrams in between.

opendis 1.0 requires numpy<2 in its metadata, but works with the numpy 2 of this repository, so install it without its
dependencies: pip install --no-deps opendis==1.0
"""
import ipaddress
import math
import socket
import struct
import time
from collections import deque
from io import BytesIO

import numpy as np

try:
    from opendis.DataOutputStream import DataOutputStream
    from opendis.PduFactory import createPdu
    from opendis.RangeCoordinates import GPS
    from opendis.dis7 import (DeadReckoningParameters, DetonationPdu, EntityID, EntityMarking, EntityStatePdu, EntityType,
                              EulerAngles, EventIdentifier, FirePdu, MunitionDescriptor, SimulationAddress, Vector3Double,
                              Vector3Float)
except ImportError as e:
    raise ImportError("The DIS connection needs opendis: pip install --no-deps opendis==1.0") from e

PROTOCOL_VERSION = 7  # sent PDUs are DIS 7, received PDUs of DIS 6 have the same layout
HEADER_LENGTH = 12

# PDU types and protocol families
ENTITY_STATE = 1
FIRE = 2
DETONATION = 3
ENTITY_STATE_UPDATE = 67
FAMILY_ENTITY_INFORMATION = 1
FAMILY_WARFARE = 2

# Dead reckoning algorithms: static, then F/R (fixed/rotating orientation), P/V (constant velocity/acceleration),
# W/B (velocity and acceleration in world/body coordinates)
DRM_STATIC, DRM_FPW, DRM_RPW, DRM_RVW, DRM_FVW, DRM_FPB, DRM_RPB, DRM_RVB, DRM_FVB = range(1, 10)
DRM_ROTATING = (DRM_RPW, DRM_RVW, DRM_RPB, DRM_RVB)
DRM_ACCELERATING = (DRM_RVW, DRM_FVW, DRM_RVB, DRM_FVB)
DRM_BODY = (DRM_FPB, DRM_RPB, DRM_RVB, DRM_FVB)

# Bits of the appearance of platforms
DAMAGE_MASK = 0b11 << 3
DAMAGE_DESTROYED = 3 << 3
SMOKE_PLUME = 1 << 5
FLAMING = 1 << 15
FROZEN = 1 << 21
POWER_PLANT_ON = 1 << 22
DEACTIVATED = 1 << 23

DETONATION_ENTITY_IMPACT = 1
NO_ENTITY = (0, 0, 0)

GPS_CONVERSIONS = GPS()  # WGS84 conversions of opendis between geodetic, ECEF coordinates and DIS Euler angles


def dis_timestamp():
    """Relative DIS timestamp: time past the hour in units of 3600 s / 2^31, least significant bit 0."""
    return (int((time.time() % 3600.0) / 3600.0 * 0x7FFFFFFF) << 1) & 0xFFFFFFFF


def euler_to_dcm(psi, theta, phi):
    """Rotation matrix from a reference frame to body axes for the Euler angles psi (yaw), theta (pitch) and phi (roll) in
    rad, rotated about z, y and x. For the DIS orientation the reference frame is ECEF, for roll/pitch/yaw it is north/east/down."""
    cps, sps = math.cos(psi), math.sin(psi)
    cth, sth = math.cos(theta), math.sin(theta)
    cph, sph = math.cos(phi), math.sin(phi)
    return np.array([
        [cth * cps, cth * sps, -sth],
        [sph * sth * cps - cph * sps, sph * sth * sps + cph * cps, sph * cth],
        [cph * sth * cps + sph * sps, cph * sth * sps - sph * cps, cph * cth],
    ])


def dcm_to_euler(dcm):
    """Euler angles (psi, theta, phi) in rad of a rotation matrix from a reference frame to body axes, see euler_to_dcm()."""
    theta = -math.asin(max(-1.0, min(1.0, dcm[0, 2])))
    return math.atan2(dcm[0, 1], dcm[0, 0]), theta, math.atan2(dcm[1, 2], dcm[2, 2])


def dr_rotation(angular_velocity, dt):
    """Rotation of the body axes during dt seconds at the angular velocity (body axes, rad/s), see IEEE 1278.1 Annex E."""
    rate = float(np.linalg.norm(angular_velocity))
    angle = rate * dt
    if abs(angle) < 1e-9:
        return np.eye(3)
    wx, wy, wz = angular_velocity / rate
    skew = np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])
    axis = np.array([wx, wy, wz])
    return (1.0 - math.cos(angle)) * np.outer(axis, axis) + math.cos(angle) * np.eye(3) - math.sin(angle) * skew


def geodetic_to_dis(lat, lon, alt, roll, pitch, yaw, velocity_ned):
    """Converts a geodetic state into DIS world coordinates.

    Args: latitude and longitude in rad, altitude in m, roll/pitch/yaw in rad, north/east/down velocity in m/s.
    Returns: ECEF location (m), ECEF velocity (m/s) and DIS Euler angles (psi, theta, phi) in rad.
    """
    x, y, z, psi, theta, phi = GPS_CONVERSIONS.llarpy2ecef(lat, lon, alt, roll, pitch, yaw)
    velocity = GPS_CONVERSIONS.ecefvec2nedvec(lat, lon).T @ np.asarray(velocity_ned, dtype=float)
    return np.array([x, y, z]), velocity, (psi, theta, phi)


def dis_to_geodetic(location, velocity, orientation):
    """Converts a state in DIS world coordinates into a geodetic state.

    Args: ECEF location (m), ECEF velocity (m/s), orientation as rotation matrix from ECEF to body axes.
    Returns: (latitude, longitude in rad, altitude in m), (roll, pitch, yaw in rad), north/east/down velocity in m/s.
    """
    lat_deg, lon_deg, alt = GPS_CONVERSIONS.ecef2lla(location)
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    ned = GPS_CONVERSIONS.ecefvec2nedvec(lat, lon)
    yaw, pitch, roll = dcm_to_euler(orientation @ ned.T)
    return (lat, lon, alt), (roll, pitch, yaw), ned @ velocity


def entity_id(record):
    """(site, application, entity) of an opendis EntityID."""
    return record.siteID, record.applicationID, record.entityID


def entity_type(record):
    """(kind, domain, country, category, subcategory, specific, extra) of an opendis EntityType."""
    return (record.entityKind, record.domain, record.country, record.category, record.subcategory, record.specific,
            record.extra)


# Names of DIS enumerations (SISO-REF-010) for the messages. A PDU only carries the numbers.
DETONATION_RESULTS = {
    0: "other", 1: "entity impact", 2: "entity proximate detonation", 3: "ground impact", 4: "ground proximate detonation",
    5: "detonation", 6: "none or no detonation (dud)", 7: "HE hit, small", 8: "HE hit, medium", 9: "HE hit, large",
    10: "armor-piercing hit", 11: "dirt blast, small", 12: "dirt blast, medium", 13: "dirt blast, large",
    14: "water blast, small", 15: "water blast, medium", 16: "water blast, large", 17: "air hit",
    18: "building hit, small", 19: "building hit, medium", 20: "building hit, large", 21: "mine-clearing line charge",
    22: "environment object impact", 23: "environment object proximate detonation", 24: "water impact", 25: "air burst",
}
MUNITION_DOMAINS = {1: "anti-air", 2: "anti-armor", 3: "anti-guided weapon", 4: "anti-radar", 5: "anti-submarine",
                    6: "anti-ship", 7: "bomb", 8: "battlefield support", 9: "strategic", 10: "tactical",
                    11: "directed energy"}
MUNITION_CATEGORIES = {1: "guided", 2: "ballistic", 3: "fixed"}
COUNTRIES = {45: "China", 71: "France", 78: "Germany", 222: "Russia", 224: "UK", 225: "USA"}


def detonation_result_name(result):
    """Name of a DIS detonation result, e.g. 1: "entity impact"."""
    return DETONATION_RESULTS.get(result, f"result {result}")


def munition_name(munition_type):
    """The DIS entity type of a munition with the names of its kind, domain, country and category, e.g.
    "2.1.225.1.2.3.0 (munition, anti-air, USA, guided)". The name of the specific munition (e.g. AIM-120C) is in the entity
    type mapping of the application (e.g. VR-Forces), not in the PDU."""
    kind, domain, country, category = munition_type[:4]
    numbers = ".".join(map(str, munition_type))
    if kind != 2:
        return f"{numbers} (kind {kind}, not a munition)"
    names = ["munition", MUNITION_DOMAINS.get(domain, f"domain {domain}"), COUNTRIES.get(country, f"country {country}"),
             MUNITION_CATEGORIES.get(category, f"category {category}")]
    return f"{numbers} ({', '.join(names)})"


def marking(record):
    """Text of an opendis EntityMarking, whose characters are read as signed bytes."""
    text = bytes(c & 0xFF for c in record.characters).split(b"\0", 1)[0]
    return text.decode("ascii", errors="replace").strip()


class RemoteEntity:
    """Entity of another application of the exercise, dead reckoned from its last entity state PDU (state_at())."""

    def __init__(self, entity_id, first_seen):
        self.id = entity_id                  # (site, application, entity)
        self.marking = ""
        self.force_id = 0
        self.type = (0,) * 7
        self.appearance = 0
        self.dr_algorithm = DRM_FPW
        self.location = np.zeros(3)          # ECEF, m
        self.velocity = np.zeros(3)          # ECEF, m/s
        self.acceleration = np.zeros(3)      # ECEF, m/s^2
        self.angular_velocity = np.zeros(3)  # body axes, rad/s
        self.orientation = np.eye(3)         # rotation matrix from ECEF to body axes
        self.first_seen = first_seen         # time.perf_counter() of the first and the last entity state PDU
        self.received = first_seen

    @property
    def is_air_platform(self):
        return self.type[0] == 1 and self.type[1] == 2

    @property
    def destroyed(self):
        return self.appearance & DAMAGE_MASK == DAMAGE_DESTROYED

    @property
    def deactivated(self):
        return bool(self.appearance & DEACTIVATED)

    def age(self, now):
        """Seconds since the last entity state PDU."""
        return now - self.received

    def update(self, pdu, received):
        """Takes the state of an entity state (update) PDU received at time received (time.perf_counter())."""
        orientation = euler_to_dcm(pdu.entityOrientation.psi, pdu.entityOrientation.theta, pdu.entityOrientation.phi)
        location = np.array([pdu.entityLocation.x, pdu.entityLocation.y, pdu.entityLocation.z])
        velocity = np.array([pdu.entityLinearVelocity.x, pdu.entityLinearVelocity.y, pdu.entityLinearVelocity.z])
        if not (np.isfinite(orientation).all() and np.isfinite(location).all() and np.isfinite(velocity).all()):
            raise ValueError(f"Invalid state of entity {self.id}")
        if pdu.pduType == ENTITY_STATE:  # an entity state update PDU only has the state, the rest is kept
            self.marking = marking(pdu.marking)
            self.force_id = pdu.forceId
            self.type = entity_type(pdu.entityType)
            dr = pdu.deadReckoningParameters
            self.dr_algorithm = dr.deadReckoningAlgorithm
            acceleration = dr.entityLinearAcceleration
            self.acceleration = np.array([acceleration.x, acceleration.y, acceleration.z])
            rates = dr.entityAngularVelocity
            self.angular_velocity = np.array([rates.x, rates.y, rates.z])
        self.appearance = pdu.entityAppearance
        self.location, self.velocity, self.orientation = location, velocity, orientation
        if self.dr_algorithm in DRM_BODY:  # velocity and acceleration in body axes
            self.velocity = orientation.T @ velocity
            if pdu.pduType == ENTITY_STATE:
                self.acceleration = orientation.T @ self.acceleration
        if self.dr_algorithm not in DRM_ACCELERATING:
            self.acceleration = np.zeros(3)
        self.received = received

    def state_at(self, now):
        """ECEF location, ECEF velocity and orientation (rotation matrix from ECEF to body axes), dead reckoned to time now."""
        if self.dr_algorithm == DRM_STATIC:
            return self.location, np.zeros(3), self.orientation
        if self.appearance & FROZEN:  # e.g. while the scenario of VR-F is paused
            return self.location, self.velocity, self.orientation
        dt = now - self.received
        location = self.location + self.velocity * dt + 0.5 * self.acceleration * dt * dt
        velocity = self.velocity + self.acceleration * dt
        orientation = self.orientation
        if self.dr_algorithm in DRM_ROTATING:
            orientation = dr_rotation(self.angular_velocity, dt) @ self.orientation
        return location, velocity, orientation


class Detonation:
    """Detonation of a munition of another application of the exercise."""

    def __init__(self, pdu, received):
        self.firing_id = entity_id(pdu.firingEntityID)
        self.target_id = entity_id(pdu.targetEntityID)
        self.location = np.array([pdu.location.x, pdu.location.y, pdu.location.z])  # ECEF, m
        self.result = pdu.detonationResult
        self.munition_type = entity_type(pdu.descriptor.munitionType)
        self.received = received


class DISComm:
    """UDP connection to a DIS exercise.

    Args:
        address: destination of the sent PDUs: the broadcast address of the network (e.g. 10.11.255.255), the IP of the computer
            of the other application, or a multicast group, which is joined to receive.
        port: UDP port of the exercise.
        exercise_id: DIS exercise ID, PDUs of other exercises are ignored.
        site_id, application_id: DIS simulation address of this program. Its own PDUs, which it receives back with broadcast,
            are ignored, so the address must differ from the ones of the other applications (e.g. VR-Forces).
        receive_port: port on which the PDUs are received, by default port.
    """

    RECEIVE_BUFFER = 4 * 1024 * 1024  # bytes, the datagrams arriving between two polls must fit
    MAX_DATAGRAM = 65535

    def __init__(self, address="10.11.255.255", port=3000, exercise_id=1, site_id=1, application_id=8492, receive_port=None):
        self.address = socket.gethostbyname(address)
        self.port = port
        self.exercise_id = exercise_id
        self.simulation_address = (site_id, application_id)
        self.entities = {}          # RemoteEntity per entity ID (site, application, entity) of the other applications
        self.detonations = deque()  # Detonations of the other applications, removed by the user of DISComm
        self.stats = {"received": 0, "sent": 0, "entity_states": 0, "fires": 0, "detonations": 0, "other": 0,
                      "receive_errors": 0, "send_errors": 0}
        self._event_number = 0
        self._readers = {ENTITY_STATE: self._read_entity_state, ENTITY_STATE_UPDATE: self._read_entity_state,
                         FIRE: self._read_fire, DETONATION: self._read_detonation}
        self._last_error = ""
        self.socket = self._open_socket(receive_port or port)

    def _open_socket(self, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        # Other DIS applications on this computer (e.g. VR-Forces) may use the same port
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.RECEIVE_BUFFER)
        except OSError:
            pass  # the default buffer of the operating system is used
        sock.bind(("", port))
        if ipaddress.ip_address(self.address).is_multicast:
            membership = struct.pack("4s4s", socket.inet_aton(self.address), socket.inet_aton("0.0.0.0"))
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
        sock.setblocking(False)
        return sock

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def poll(self, max_datagrams=10000):
        """Reads the PDUs received since the last call, without waiting. @returns the number of datagrams."""
        now = time.perf_counter()
        count = 0
        for _ in range(max_datagrams):
            try:
                data, _ = self.socket.recvfrom(self.MAX_DATAGRAM)
            except BlockingIOError:
                break
            except ConnectionResetError:
                continue  # Windows reports here that an earlier datagram could not be delivered (ICMP port unreachable)
            except OSError as e:
                self.stats["receive_errors"] += 1
                self._last_error = f"receive: {e}"
                break
            count += 1
            self._read_datagram(data, now)
        self.stats["received"] += count
        return count

    def _read_datagram(self, data, now):
        """Reads the PDUs of a datagram, which can contain several (bundled) PDUs."""
        offset = 0
        while len(data) - offset >= HEADER_LENGTH:
            exercise_id, pdu_type = data[offset + 1], data[offset + 2]
            (length,) = struct.unpack_from(">H", data, offset + 8)
            if length < HEADER_LENGTH or offset + length > len(data):
                self.stats["other"] += 1
                return
            if exercise_id == self.exercise_id and pdu_type in self._readers:
                try:
                    self._readers[pdu_type](data[offset:offset + length], now)
                except Exception as e:  # e.g. struct.error of opendis for a truncated PDU, which must not stop the reception
                    self.stats["other"] += 1
                    self._last_error = f"PDU type {pdu_type}: {e!r}"
            offset += length
            # DIS 7 aligns bundled PDUs at 8 bytes
            if offset % 8 and not self._is_header(data, offset):
                offset += 8 - offset % 8

    @staticmethod
    def _is_header(data, offset):
        return (len(data) - offset >= HEADER_LENGTH and 1 <= data[offset] <= PROTOCOL_VERSION
                and struct.unpack_from(">H", data, offset + 8)[0] >= HEADER_LENGTH)

    def _read_entity_state(self, data, now):
        key = struct.unpack_from(">HHH", data, HEADER_LENGTH)
        if key[:2] == self.simulation_address:
            return  # own PDU, received back with broadcast
        pdu = createPdu(data)
        entity = self.entities.get(key)
        if entity is None:
            if pdu.pduType != ENTITY_STATE:
                return  # the type and marking of an entity come with its first entity state PDU
            entity = self.entities[key] = RemoteEntity(key, now)
        entity.update(pdu, now)
        self.stats["entity_states"] += 1

    def _read_fire(self, data, now):
        if struct.unpack_from(">HH", data, HEADER_LENGTH) != self.simulation_address:
            self.stats["fires"] += 1

    def _read_detonation(self, data, now):
        if struct.unpack_from(">HH", data, HEADER_LENGTH) == self.simulation_address:
            return  # own detonation, received back with broadcast
        self.detonations.append(Detonation(createPdu(data), now))
        self.stats["detonations"] += 1

    def new_event_number(self):
        """Number of a new fire event, used again by its detonation."""
        self._event_number = self._event_number % 0xFFFF + 1
        return self._event_number

    def send_entity_state(self, entity_number, force_id, entity_type, marking, location, velocity, orientation,
                          angular_velocity, appearance):
        """Sends the state of an entity of this program, dead reckoned by the receivers with position, velocity, orientation
        and angular velocity (DRM_RPW).

        Args:
            entity_number: entity of this program, its ID is (site_id, application_id, entity_number).
            force_id: DIS force, e.g. 1 friendly, 2 opposing.
            entity_type: DIS entity type (kind, domain, country, category, subcategory, specific, extra).
            marking: name of up to 11 ASCII characters, e.g. shown by VR-Forces.
            location, velocity: ECEF in m and m/s.
            orientation: DIS Euler angles (psi, theta, phi) in rad.
            angular_velocity: body axes in rad/s.
            appearance: DIS appearance bits, e.g. DAMAGE_DESTROYED.
        """
        pdu = EntityStatePdu()
        self._header(pdu, ENTITY_STATE, FAMILY_ENTITY_INFORMATION)
        pdu.entityID = EntityID(*self.simulation_address, entity_number)
        pdu.forceId = force_id
        pdu.entityType = EntityType(*entity_type)
        pdu.alternativeEntityType = EntityType(*entity_type)
        pdu.entityLinearVelocity = Vector3Float(*map(float, velocity))
        pdu.entityLocation = Vector3Double(*map(float, location))
        pdu.entityOrientation = EulerAngles(*map(float, orientation))
        pdu.entityAppearance = appearance
        pdu.deadReckoningParameters = DeadReckoningParameters(DRM_RPW, [0] * 15, Vector3Float(),
                                                              Vector3Float(*map(float, angular_velocity)))
        pdu.marking = EntityMarking(characterSet=1)  # ASCII
        pdu.marking.setString(marking.encode("ascii", errors="replace").decode("ascii")[:11])
        pdu.capabilities = 0
        return self._send(pdu)

    def send_fire(self, entity_number, target_id, event_number, location, velocity, munition, range_):
        """Sends the firing of a munition by an entity of this program.

        Args:
            entity_number: firing entity of this program.
            target_id: (site, application, entity) of the target, NO_ENTITY if unknown.
            event_number: from new_event_number(), used again by the detonation of the munition.
            location, velocity: ECEF location (m) and velocity (m/s) of the munition when fired.
            munition: (entity type, warhead, fuse) of the munition, DIS enumerations.
            range_: m, range to the target, 0 if unknown.
        """
        pdu = FirePdu()
        self._header(pdu, FIRE, FAMILY_WARFARE)
        pdu.firingEntityID = EntityID(*self.simulation_address, entity_number)
        pdu.targetEntityID = EntityID(*target_id)
        pdu.munitionExpendableID = EntityID()  # the munition is not simulated as entity
        pdu.eventID = EventIdentifier(SimulationAddress(*self.simulation_address), event_number)
        pdu.location = Vector3Double(*map(float, location))
        pdu.descriptor = self._munition_descriptor(munition)
        pdu.velocity = Vector3Float(*map(float, velocity))
        pdu.range = float(range_)
        return self._send(pdu)

    def send_detonation(self, entity_number, target_id, event_number, location, velocity, munition,
                        result=DETONATION_ENTITY_IMPACT):
        """Sends the detonation of a munition fired by an entity of this program (send_fire() with the same event number).

        Args:
            location, velocity: ECEF location (m) and velocity (m/s) of the munition when it detonated.
            result: DIS detonation result, by default entity impact (a hit of the target).
            The other arguments are as in send_fire().
        """
        pdu = DetonationPdu()
        self._header(pdu, DETONATION, FAMILY_WARFARE)
        pdu.firingEntityID = EntityID(*self.simulation_address, entity_number)
        pdu.targetEntityID = EntityID(*target_id)
        pdu.explodingEntityID = EntityID()
        pdu.eventID = EventIdentifier(SimulationAddress(*self.simulation_address), event_number)
        pdu.velocity = Vector3Float(*map(float, velocity))
        pdu.location = Vector3Double(*map(float, location))
        pdu.descriptor = self._munition_descriptor(munition)
        pdu.locationInEntityCoordinates = Vector3Float()
        pdu.detonationResult = result
        return self._send(pdu)

    @staticmethod
    def _munition_descriptor(munition):
        munition_type, warhead, fuse = munition
        return MunitionDescriptor(EntityType(*munition_type), warhead, fuse, quantity=1, rate=0)

    def _header(self, pdu, pdu_type, family):
        # opendis initializes the PDU type and family of the instances with 0, and the PDU status with an object
        pdu.protocolVersion = PROTOCOL_VERSION
        pdu.exerciseID = self.exercise_id
        pdu.pduType = pdu_type
        pdu.protocolFamily = family
        pdu.timestamp = dis_timestamp()
        pdu.pduStatus = 0

    def _send(self, pdu):
        stream = BytesIO()
        pdu.serialize(DataOutputStream(stream))
        data = bytearray(stream.getvalue())
        struct.pack_into(">H", data, 8, len(data))  # PDU length, opendis writes its length attribute, which is 0
        try:
            self.socket.sendto(data, (self.address, self.port))
        except OSError as e:
            self.stats["send_errors"] += 1
            self._last_error = f"send: {e}"
            return False
        self.stats["sent"] += 1
        return True

    def pop_error(self):
        """@returns the last error since the last call, or an empty string."""
        error, self._last_error = self._last_error, ""
        return error

    def entity_table(self, now=None):
        """@returns a text table of the received entities."""
        now = time.perf_counter() if now is None else now
        lines = [f"{'ID':>16}  {'marking':11}  {'force':>5}  {'type':19}  {'lat':>8}  {'lon':>8}  {'alt':>6}  {'speed':>5}  "
                 f"{'age':>5}  state"]
        for entity in sorted(self.entities.values(), key=lambda e: e.first_seen):
            location, velocity, _ = entity.state_at(now)
            lat, lon, alt = GPS_CONVERSIONS.ecef2lla(location)
            state = "deactivated" if entity.deactivated else "destroyed" if entity.destroyed else "active"
            lines.append(f"{':'.join(map(str, entity.id)):>16}  {entity.marking:11}  {entity.force_id:5d}  "
                         f"{'.'.join(map(str, entity.type)):19}  {lat:8.4f}  {lon:8.4f}  {alt:6.0f}  "
                         f"{np.linalg.norm(velocity):5.0f}  {entity.age(now):5.1f}  {state}")
        return "\n".join(lines)


def list_entities(interval=2.0, **connection):
    """Prints the entities of the exercise every interval seconds until Ctrl+C, e.g. to find the markings of VR-F entities.
    connection: the arguments of DISComm."""
    dis = DISComm(**connection)
    print(f"Listening to DIS exercise {dis.exercise_id} on port {dis.port}, Ctrl+C to stop.")
    last_print = time.perf_counter()
    try:
        while True:
            dis.poll()
            if time.perf_counter() - last_print >= interval:
                last_print = time.perf_counter()
                print(f"\n{len(dis.entities)} entities, {dis.stats['received']} datagrams received:\n{dis.entity_table()}")
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        dis.close()
