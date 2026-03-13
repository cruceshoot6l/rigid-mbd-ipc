#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Regression for independently assembled tangential candidates
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu

from generalContactPotentialFriction import run_case


def run_regression():
    ipc = run_case(exu.ContactFormulation.IPCBarrier, enable_friction=True)
    gcp = run_case(exu.ContactFormulation.GCPBarrier, enable_friction=True)
    ogc = run_case(exu.ContactFormulation.OGCBarrier, enable_friction=True)

    for name, case in [("IPC", ipc), ("GCP", gcp), ("OGC", ogc)]:
        if case["contactCandidates"] <= 0:
            raise ValueError(f"{name} tangential split regression expected active normal candidates")
        if case["tangentialCandidates"] <= 0:
            raise ValueError(f"{name} tangential split regression expected active tangential candidates")
        if case["tangentialCandidates"] >= case["contactCandidates"]:
            raise ValueError(f"{name} tangential split regression expected tangential candidates to be fewer than normal candidates")
        if case["vfCandidates"] + case["eeCandidates"] != case["contactCandidates"]:
            raise ValueError(f"{name} tangential split regression returned inconsistent normal candidate accounting")

    if ogc["tangentialCandidates"] >= gcp["tangentialCandidates"]:
        raise ValueError("OGC tangential split regression expected OGC to keep a stricter tangential candidate set than GCP")

    return {"ipc": ipc, "gcp": gcp, "ogc": ogc}


def main():
    results = run_regression()
    ipc = results["ipc"]
    gcp = results["gcp"]
    ogc = results["ogc"]
    test_value = float(
        ipc["contactCandidates"] + gcp["contactCandidates"] + ogc["contactCandidates"]
        + ipc["tangentialCandidates"] + gcp["tangentialCandidates"] + ogc["tangentialCandidates"]
        + gcp["vfCandidates"] + ogc["eeCandidates"]
    )

    exu.Print("generalContactPotentialTangentialSplit=", test_value)


if __name__ == "__main__":
    main()
