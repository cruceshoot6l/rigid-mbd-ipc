#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Week-6 mechanism-like benchmark for an articulated arm stop contact
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu

from generalContactBenchmarkCommon import run_single_benchmark


results = run_single_benchmark("articulated_arm_stop")
penalty = results[0]
gcp = results[1]

if gcp["maxPenetration"] >= penalty["maxPenetration"]:
    raise ValueError("expected GCP barrier to reduce articulated-arm penetration")
if gcp["newtonStepsCount"] <= penalty["newtonStepsCount"]:
    raise ValueError("expected barrier arm benchmark to exercise a more nonlinear solve")

test_value = penalty["maxPenetration"] + gcp["maxPenetration"] + 0.01 * gcp["newtonStepsCount"]
exu.Print("generalContactBenchmarkArticulatedArm=", test_value)
