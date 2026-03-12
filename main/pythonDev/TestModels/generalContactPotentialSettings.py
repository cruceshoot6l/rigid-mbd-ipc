#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Minimal configuration test for week-1 GeneralContact potential settings
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu

SC = exu.SystemContainer()
mbs = SC.AddSystem()

gContact = mbs.AddGeneralContact()
gContact.contactFormulation = exu.ContactFormulation.GCPBarrier
gContact.barrierActivationDistance = 2.5e-3
gContact.barrierStiffness = 3.0
gContact.barrierMinimumDistance = 1e-7
gContact.useNonlinearCCDStepFilter = True
gContact.ccdTolerance = 1e-5
gContact.useGaussNewtonHessian = False

pyObject = gContact.GetPythonObject()

testValue = (
    int(gContact.contactFormulation)
    + pyObject["barrierActivationDistance"]
    + pyObject["barrierStiffness"]
    + pyObject["barrierMinimumDistance"]
    + pyObject["ccdTolerance"]
    + int(pyObject["useNonlinearCCDStepFilter"])
    + int(pyObject["useGaussNewtonHessian"])
)

exu.Print("generalContactPotentialSettings=", testValue)
