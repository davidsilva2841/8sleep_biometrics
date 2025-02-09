'''
# Original Author:

- **Paul van Gent**
- [HeartPy on PyPI](https://pypi.org/project/heartpy/)
- [GitHub Repository](https://github.com/paulvangentcom/heartrate_analysis_python)
- [Heart Rate Analysis for Human Factors: Development and Validation of an Open-Source Toolkit for Noisy Naturalistic Heart Rate Data](https://www.researchgate.net/publication/325967542_Heart_Rate_Analysis_for_Human_Factors_Development_and_Validation_of_an_Open_Source_Toolkit_for_Noisy_Naturalistic_Heart_Rate_Data)
- [Analysing Noisy Driver Physiology in Real-Time Using Off-the-Shelf Sensors: Heart Rate Analysis Software from the Taking the Fast Lane Project](https://www.researchgate.net/publication/328654252_Analysing_Noisy_Driver_Physiology_Real-Time_Using_Off-the-Shelf_Sensors_Heart_Rate_Analysis_Software_from_the_Taking_the_Fast_Lane_Project?channel=doi&linkId=5bdab2c84585150b2b959d13&showFulltext=true)

Custom exceptions and warnings for HeartPy
'''

__all__ = ['BadSignalWarning']

class BadSignalWarning(UserWarning):
    '''
    warning class to raise when no heart rate is detectable
    in supplied signal.

    This warning notifies the user that the supplied signal is
    of insufficient quality and/or does not contain enough information
    to properly process.
    '''

class IncorrectFileType(UserWarning):
    '''
    warning class to raise when incorrect file type or incorrectly
    formatted file is provided to data loader
    '''
