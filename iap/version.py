class Version:
    def __init__(self, version_str):
        split_version = version_str.split("-")
        domain, version_num = split_version[:2]
        self.extra = "-".join(split_version[2:])
        self.domain = domain
        self.version_num = float(version_num)

    def __lt__(self, ver2):
        return (self.domain == ver2.domain) and (self.version_num < ver2.version_num)

    def __gt__(self, ver2):
        return (self.domain == ver2.domain) and (self.version_num > ver2.version_num)

    def __eq__(self, ver2):
        return (self.domain == ver2.domain) and (self.version_num == ver2.version_num)

    def __str__(self):
        if self.extra:
            return f"{self.domain}-{self.version_num}-{self.extra}"
        return f"{self.domain}-{self.version_num}"


def parse_version(version_str):
    return Version(version_str)
