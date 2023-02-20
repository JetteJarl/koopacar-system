def header_to_float_stamp(header):
    """Converts the stamp of header to float."""
    return float(f"{header.stamp.sec}.{header.stamp.nanosec}")
