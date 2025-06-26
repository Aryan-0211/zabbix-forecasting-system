import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json # ADDED: For explicit JSON handling in error messages (though might be implicitly imported by requests)

load_dotenv()

class ZabbixConnector:
    def __init__(self):
        self.api_url = os.getenv("ZABBIX_API_URL")
        self.token = os.getenv("ZABBIX_API_TOKEN")
        self.debug = True
        self.item_cache = {}

    def _log(self, message):
        if self.debug:
            print(f"[ZabbixConnector] {message}")

    def fetch_metric(self, itemid, hours=None, time_from=None, time_till=None):
        """
        Fetch metric data from Zabbix for a specified duration or time range.
        Provide either 'hours' OR ('time_from' and 'time_till').
        """
        self._log(f"Fetching itemid={itemid}")

        payload = {
            "jsonrpc": "2.0",
            "method": "history.get",
            "params": {
                "output": "extend",
                "itemids": itemid,
                "history": 0,  # 0=numeric values
                "sortfield": "clock",
                "sortorder": "ASC",
            },
            "auth": self.token,
            "id": 1
        }

        if hours is not None:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            payload["params"]["time_from"] = int(start_time.timestamp())
            payload["params"]["time_till"] = int(end_time.timestamp())
            self._log(f"  Range: last {hours} hours (from {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')})")
        elif time_from is not None and time_till is not None:
            # Ensure time_from/time_till are datetime objects or can be converted
            if isinstance(time_from, pd.Timestamp): time_from = time_from.to_pydatetime()
            if isinstance(time_till, pd.Timestamp): time_till = time_till.to_pydatetime()

            payload["params"]["time_from"] = int(time_from.timestamp())
            payload["params"]["time_till"] = int(time_till.timestamp())
            self._log(f"  Range: specific (from {time_from.strftime('%Y-%m-%d %H:%M')} to {time_till.strftime('%Y-%m-%d %H:%M')})")
        else:
            raise ValueError("Must provide either 'hours' or both 'time_from' and 'time_till'.")

        try:
            response = requests.post(self.api_url, json=payload, timeout=10)
            response.raise_for_status()
            
            # --- DEBUGGING LINE ADDED HERE ---
            print(f"[ZabbixConnector] Raw fetch_metric response text: {response.text[:500]}...") # Print first 500 chars
            # --- END DEBUGGING LINE ---

            json_data = response.json() # Error might still occur here if not JSON
            
            if "error" in json_data:
                error = json_data["error"]
                raise ValueError(f"Zabbix API Error: {error['data']} (Code: {error.get('code')})")
                
            if not json_data.get("result"):
                self._log("Empty response from Zabbix API. No data found for the requested period.")
                return pd.DataFrame(columns=["timestamp", "value"])

            records = []
            for item in json_data["result"]:
                records.append({
                    "timestamp": datetime.fromtimestamp(int(item["clock"])),
                    "value": float(item["value"])
                })

            df = pd.DataFrame(records)
            self._log(f"Retrieved {len(df)} records. Columns: {list(df.columns)}")
            if not df.empty:
                self._log(f"Sample data:\n{df.head(3)}")

            return df

        except requests.exceptions.Timeout:
            self._log(f"Error fetching data: Connection to Zabbix API timed out: {self.api_url}")
            raise ConnectionError(f"Connection to Zabbix API timed out: {self.api_url}")
        except requests.exceptions.RequestException as e:
            self._log(f"Error fetching data: Request failed: {e}")
            raise ConnectionError(f"Request to Zabbix API failed: {e}")
        except json.decoder.JSONDecodeError as e:
            self._log(f"Error fetching data: Invalid JSON response from {self.api_url}. Response text: {response.text[:500]}... Error: {e}") # Increased text limit
            raise ValueError(f"Invalid JSON response from Zabbix API: {e}")
        except Exception as e:
            self._log(f"An unexpected error occurred while fetching data: {e}")
            raise Exception(f"An unexpected error occurred: {e}")

    def get_item_id(self, host_name, item_key):
        # Check cache first
        cache_key = f"{host_name}-{item_key}"
        if cache_key in self.item_cache:
            self._log(f"Using cached itemid for {host_name}/{item_key}")
            return self.item_cache[cache_key]

        self._log(f"Searching for item: {item_key} on {host_name}")
        host_payload = {
            "jsonrpc": "2.0",
            "method": "host.get",
            "params": {
                "output": ["hostid"],
                "filter": {"name": host_name}
            },
            "auth": self.token,
            "id": 1
        }

        try:
            host_response_raw = requests.post(self.api_url, json=host_payload, timeout=10) # Added timeout
            # --- DEBUGGING LINE ADDED HERE ---
            print(f"[ZabbixConnector] Raw host.get response text: {host_response_raw.text[:500]}...") # Print first 500 chars
            # --- END DEBUGGING LINE ---
            host_response_raw.raise_for_status() # Check for HTTP errors before json()
            host_response = host_response_raw.json() # Error might occur here
            
            if "error" in host_response:
                error_msg = host_response["error"].get("data", host_response["error"].get("message", "Unknown Zabbix API error"))
                raise ValueError(f"Zabbix API Error (Host Get): {error_msg} (Code: {host_response['error'].get('code')})") # Added code to error msg
            
            if not host_response.get("result"):
                raise ValueError(f"Host '{host_name}' not found or no result from host.get API.")

            hostid = host_response["result"][0]["hostid"]

            item_payload = {
                "jsonrpc": "2.0",
                "method": "item.get",
                "params": {
                    "output": ["itemid"],
                    "hostids": hostid,
                    "filter": {"key_": item_key},
                    "searchWildcardsEnabled": False
                },
                "auth": self.token,
                "id": 2
            }

            item_response_raw = requests.post(self.api_url, json=item_payload, timeout=10) # Added timeout
            # --- DEBUGGING LINE ADDED HERE ---
            print(f"[ZabbixConnector] Raw item.get response text: {item_response_raw.text[:500]}...") # Print first 500 chars
            # --- END DEBUGGING LINE ---
            item_response_raw.raise_for_status() # Check for HTTP errors before json()
            item_response = item_response_raw.json() # Error occurs here in your traceback
            print(f"[DEBUG] Item lookup response JSON: {item_response}") # This will print if json() succeeds

            if "error" in item_response:
                error_msg = item_response["error"].get("data", item_response["error"].get("message", "Unknown Zabbix API error"))
                raise ValueError(f"Zabbix API Error (Item Get): {error_msg} (Code: {item_response['error'].get('code')})") # Added code to error msg

            if not item_response.get("result"):
                raise ValueError(f"Item '{item_key}' not found on host '{host_name}' (hostid: {hostid}) or no result from item.get API.")

            itemid = item_response["result"][0]["itemid"]
            self.item_cache[cache_key] = itemid # Cache the item ID
            self._log(f"Found itemid {itemid} for {host_name}/{item_key}")
            return itemid

        except requests.exceptions.Timeout as timeout_exc:
            print(f"[ERROR] Zabbix API request timed out: {timeout_exc}")
            raise ConnectionError(f"Connection to Zabbix API timed out during item ID lookup: {timeout_exc}")
        except requests.exceptions.RequestException as req_exc:
            # Catch network/request-specific errors like connection issues or bad HTTP status codes
            print(f"[ERROR] Zabbix API request failed (HTTP/Network): {req_exc}")
            if 'host_response_raw' in locals() and host_response_raw:
                print(f"  Host response status: {host_response_raw.status_code}, text: {host_response_raw.text[:500]}...")
            if 'item_response_raw' in locals() and item_response_raw:
                print(f"  Item response status: {item_response_raw.status_code}, text: {item_response_raw.text[:500]}...")
            raise ConnectionError(f"Request to Zabbix API failed: {req_exc}")
        except json.decoder.JSONDecodeError as json_exc:
            # Catch JSON decoding errors specifically
            print(f"[ERROR] Zabbix API returned non-JSON response. Details: {json_exc}")
            print(f"  Attempted URL: {self.api_url}")
            # Ensure host_response_raw/item_response_raw are defined before accessing .text
            if 'host_response_raw' in locals() and host_response_raw:
                print(f"  Host response text (first 500 chars): {host_response_raw.text[:500]}...")
            if 'item_response_raw' in locals() and item_response_raw:
                print(f"  Item response text (first 500 chars): {item_response_raw.text[:500]}...")
            raise ValueError(f"Invalid JSON response from Zabbix API: {json_exc}")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during item ID lookup: {str(e)}")
            raise

    def get_available_metrics(self, host_name):
        try:
            host_payload = {
                "jsonrpc": "2.0",
                "method": "host.get",
                "params": {
                    "output": ["hostid"],
                    "filter": {"name": host_name}
                },
                "auth": self.token,
                "id": 1
            }

            host_response_raw = requests.post(self.api_url, json=host_payload, timeout=10) # Added timeout
            # --- DEBUGGING LINE ADDED HERE ---
            print(f"[ZabbixConnector] Raw get_available_metrics host.get response text: {host_response_raw.text[:500]}...")
            # --- END DEBUGGING LINE ---
            host_response_raw.raise_for_status() # Check for HTTP errors
            host_response = host_response_raw.json()
            
            if "error" in host_response:
                error_msg = host_response["error"].get("data", host_response["error"].get("message", "Unknown Zabbix API error"))
                raise ValueError(f"Zabbix API Error (Host Get for Metrics): {error_msg} (Code: {host_response['error'].get('code')})")
            
            if not host_response.get("result"):
                raise ValueError(f"Host '{host_name}' not found for getting metrics.")

            hostid = host_response["result"][0]["hostid"]

            item_payload = {
                "jsonrpc": "2.0",
                "method": "item.get",
                "params": {
                    "output": ["itemid", "name", "key_"],
                    "hostids": hostid,
                    "searchWildcardsEnabled": False
                },
                "auth": self.token,
                "id": 2
            }

            item_response_raw = requests.post(self.api_url, json=item_payload, timeout=10) # Added timeout
            # --- DEBUGGING LINE ADDED HERE ---
            print(f"[ZabbixConnector] Raw get_available_metrics item.get response text: {item_response_raw.text[:500]}...")
            # --- END DEBUGGING LINE ---
            item_response_raw.raise_for_status() # Check for HTTP errors
            item_response = item_response_raw.json()

            if "error" in item_response:
                error_msg = item_response["error"].get("data", item_response["error"].get("message", "Unknown Zabbix API error"))
                raise ValueError(f"Zabbix API Error (Item Get for Metrics): {error_msg} (Code: {item_response['error'].get('code')})")

            return item_response.get("result", [])

        except requests.exceptions.Timeout as timeout_exc:
            print(f"[ERROR] Zabbix API request (get_available_metrics) timed out: {timeout_exc}")
            return []
        except requests.exceptions.RequestException as req_exc:
            print(f"[ERROR] Zabbix API request (get_available_metrics) failed (HTTP/Network): {req_exc}")
            if 'host_response_raw' in locals() and host_response_raw:
                print(f"  Host response status: {host_response_raw.status_code}, text: {host_response_raw.text[:500]}...")
            if 'item_response_raw' in locals() and item_response_raw:
                print(f"  Item response status: {item_response_raw.status_code}, text: {item_response_raw.text[:500]}...")
            return []
        except json.decoder.JSONDecodeError as json_exc:
            print(f"[ERROR] Zabbix API (get_available_metrics) returned non-JSON response. Details: {json_exc}")
            if 'host_response_raw' in locals() and host_response_raw:
                print(f"  Host response text (first 500 chars): {host_response_raw.text[:500]}...")
            if 'item_response_raw' in locals() and item_response_raw:
                print(f"  Item response text (first 500 chars): {item_response_raw.text[:500]}...")
            return []
        except Exception as e:
            self._log(f"Error getting metrics: {str(e)}")
            return []
