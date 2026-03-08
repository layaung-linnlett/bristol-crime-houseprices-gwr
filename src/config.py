{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "684ce67f-718f-4e99-afb8-92ae7723ec14",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"Configuration and paths for Bristol crime-house price analysis.\"\"\"\n",
    "\n",
    "from pathlib import Path\n",
    "\n",
    "# Base directories\n",
    "BASE_PATH = Path.cwd()\n",
    "RAW_DATA_PATH = BASE_PATH / 'data' / 'raw' \n",

    "house_prices_raw_path = RAW_DATA_PATH / 'house_prices'\n",
    "crime_raw_path        = RAW_DATA_PATH / 'crime'\n",
    "geo_path              = RAW_DATA_PATH / 'geo' \n",
    "processed_data_path   = BASE_PATH / 'data' / 'processed'\n",
    "output_path           = BASE_PATH / 'output'\n",
    "\n",
    "\n",
    "# Create directories\n",
    "for path in [processed_data_path, output_path]:\n",
    "    path.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "# Analysis parameters\n",
    "CONFIG = {\n",
    "    \"scope\": {\n",
    "        \"startyear\": 2021,\n",
    "        \"endyear\": 2024,\n",
    "    },\n",
    "    \n",
    "    \"geographic\": {\n",
    "        \"bristol_postcode_prefix\": \"BS\",\n",
    "        \"bristol_lsoa_prefix\": \"E01014\",\n",
    "        \"bristol_district\": \"CITY OF BRISTOL\",\n",
    "        \"latmin\": 51.42,\n",
    "        \"latmax\": 51.50,\n",
    "        \"lonmin\": -2.649,\n",
    "        \"lonmax\": -2.530,\n",
    "    },\n",
    "    \n",
    "    \"statistical\": {\n",
    "        \"significance_level\": 0.05,\n",
    "        \"iqr_multiplier\": 1.5,\n",
    "    },\n",
    "    \n",
    "    \"gwr\": {\n",
    "        \"kernel\": \"bisquare\",\n",
    "        \"adaptive\": True,\n",
    "    },\n",
    "}"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
