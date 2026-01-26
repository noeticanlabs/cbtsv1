class IntrinsicBinder:
    def __init__(self):
        self.bound_intrinsics = {}

    def bind_intrinsic(self, name, kernel_func):
        self.bound_intrinsics[name] = kernel_func

    def is_bound(self, name):
        return name in self.bound_intrinsics

    def get_kernel(self, name):
        return self.bound_intrinsics.get(name)