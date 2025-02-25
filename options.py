# ------------------Import Libraries ------------#
from extract_options_chain import ExtractOptionsChain
from extract_options_data import ExtractOptionsData

# ---------------------CONSTANTS------------------#

# --------------------MAIN CODE-------------------#

options_data = ExtractOptionsData()
symbols = options_data.extract_available_option_symbols()
print(symbols)


