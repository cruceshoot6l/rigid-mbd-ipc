#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Week-6 mechanism-like benchmark for a guided slider against a stop
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu

from generalContactBenchmarkCommon import run_single_benchmark


results = run_single_benchmark("guided_slider_stop")
penalty = results[0]
gcp = results[1]

if gcp["maxPenetration"] >= penalty["maxPenetration"]:
    raise ValueError("expected GCP barrier to reduce guided-slider penetration")
if gcp["newtonStepsCount"] <= 0:
    raise ValueError("unexpected missing Newton iterations in guided-slider benchmark")

test_value = penalty["maxPenetration"] + gcp["maxPenetration"] + 0.01 * gcp["newtonStepsCount"]
exu.Print("generalContactBenchmarkGuidedSlider=", test_value)
