# Global variables for multiprocessing so each process doesn't have to reload the data
from data_manager import DataManager

alina = DataManager('alina', load=True)
# david = DataManager('david', load=True)
elisa = DataManager('elisa', load=True)
den = DataManager('den', load=True)
tally = DataManager('tally', load=True)
trinity = DataManager('trinity', load=True)
# data_managers = [alina, david, den, elisa, trinity, tally]
data_managers = [alina, den, elisa, trinity, tally]

