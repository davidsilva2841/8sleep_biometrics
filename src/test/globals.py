from data_manager import DataManager

david = DataManager('david', load=True)
den = DataManager('den', load=True)
tally = DataManager('tally', load=True)
trinity = DataManager('trinity', load=True)
#  TODO - FIX DEN VALIDATION DATA
data_managers = [david, tally, trinity, den]
