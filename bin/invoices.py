import json

import os


if __name__ == '__main__':
    # os.system("ansible-playbook ln.yml")
    with open("../data/listinvoices.json", "r") as list_inv:
        _inv = json.load(list_inv)
        invoices = _inv['invoices']
        pollofeed_msats = [inv['msatoshi'] for inv in invoices]
        msats = sum(pollofeed_msats)
        print("Invoices = {}".format(len(invoices)))
        print("btc = {}".format(msats / 1e11))
