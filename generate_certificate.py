import logging
import ipaddress
from cryptography import x509
from cryptography.x509.oid import NameOID
import datetime
from cryptography.hazmat.primitives import hashes, serialization

logger = logging.getLogger(f"{__name__}")


def generate_self_signed_cert(
    hostname,
    node,
    ip_addresses=None,
    country="US",
    state="California",
    locality="San Francisco",
    org_name="My Organization",
    validity_days=365,
):
    """
    Generate a self-signed certificate for the given hostname


    Args:
        hostname: The hostname for the certificate
        node: Node object w/ required key attributes
        ip_addresses: Optional list of IP addresses to include in the SAN
        country: Country name for the certificate
        state: State name for the certificate
        locality: Locality name for the certificate
        org_name: Organization name for the certificate
        validity_days: Number of data the certificate should be valid
    """
    key = node.read_private_key()
    if key is None:
        raise ValueError("Failed to read private key")

    san = [x509.DNSName(hostname)]
    if ip_addresses:
        for ip in ip_addresses:
            try:
                ip_obj = ipaddress.ip_address(ip)
                san.append(x509.IPAddress(ip_obj))
                logger.info(f"IP address {ip} added to subject alternative name list")
            except ValueError as e:
                logger.error(f"Invalid IP address {ip}: {e}")
                raise ValueError(f"Invalid IP address provided: {ip}")

    name_attributes = [
        x509.NameAttribute(NameOID.COUNTRY_NAME, country),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, state),
        x509.NameAttribute(NameOID.LOCALITY_NAME, locality),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, org_name),
        x509.NameAttribute(NameOID.COMMON_NAME, hostname),
    ]

    subject = issuer = x509.Name(name_attributes)

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=validity_days)
        )
        .add_extension(
            x509.SubjectAlternativeName(san),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    logger.info("Self-signed certficate generated sucessfully")
    return cert
