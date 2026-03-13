#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Week-6 master benchmark runner and result writer
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu

from generalContactBenchmarkCommon import (
    benchmark_signature,
    run_all_benchmarks,
    write_results_files,
    write_summary_file,
)


results = run_all_benchmarks()
write_results_files(results)
write_summary_file(results)

test_value = benchmark_signature(results)
exu.Print("generalContactBenchmarkWeek6=", test_value)
