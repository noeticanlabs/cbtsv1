class NSCError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code
        self.msg = msg

E_EMPTY_INPUT = 1
E_UNKNOWN_GLYPH = 2
E_NONCANONICAL_UNICODE = 3
E_EXPORT_IO = 4
E_CLI_USAGE = 5
E_BUNDLE_IO = 6
E_VERIFY_FAIL = 7
E_ZIP_NONDETERMINISTIC = 8
E_MANIFEST_SCHEMA = 9
E_CACHE_IO = 10
E_CACHE_INVALID = 11
E_EXPLAIN_OOB = 12
E_EXPLAIN_IO = 13