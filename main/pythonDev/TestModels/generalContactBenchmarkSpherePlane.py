#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Week-6 benchmark for sphere-plane compression
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu

from generalContactBenchmarkCommon import run_single_benchmark


results = run_single_benchmark("sphere_plane")
penalty = results[0]
gcp = results[1]

if gcp["maxPenetration"] >= penalty["maxPenetration"]:
    raise ValueError("expected GCP barrier to reduce sphere-plane penetration")
if gcp["totalPotentialCCDStepFailures"] != 0:
    raise ValueError("unexpected nonlinear CCD failure in sphere-plane benchmark")

test_value = penalty["maxPenetration"] + gcp["maxPenetration"] + 0.01 * gcp["newtonStepsCount"]
exu.Print("generalContactBenchmarkSpherePlane=", test_value)
