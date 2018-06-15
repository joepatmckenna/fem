def increment_version_number():
    with open('version', 'r') as f:
        version = f.read()
    version = [int(v) for v in version.split('.')]
    base = [10, 10, 100]
    i = 2
    version[i] += 1
    while version[i] == base[i]:
        i -= 1
        version[i] += 1
    for j in range(i + 1, 3):
        version[j] = 0
    version = '.'.join([str(v) for v in version])
    with open('version', 'w') as f:
        f.write(version)

    with open('./binder/runtime.txt', 'w') as f:
        import sys
        pyversion = sys.version_info
        f.write('%i.%i.%i' % (pyversion.major, pyversion.minor,
                              pyversion.micro))

    with open('./binder/requirements.txt', 'w') as f:
        import numpy as np
        f.write('numpy==%s\n' % (np.__version__, ))
        import matplotlib as mpl
        f.write('matplotlib==%s\n' % (mpl.__version__, ))
        # f.write('fem==%s\n' % (version, ))
        # f.write('.')


if __name__ == '__main__':
    increment_version_number()
