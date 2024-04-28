import sys, errno, re, json, ssl
import os
from urllib import request
from urllib.error import HTTPError
from time import sleep
import argparse
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_pfam_entry(base_url: str, download_file: str):
    # disable SSL verification to avoid config issues
    context = ssl._create_unverified_context()
    HEADER_SEPARATOR = "|"
    LINE_LENGTH = 80

    next = base_url

    with open(download_file, "w") as f:

        attempts = 0
        while next:
            try:
                req = request.Request(next, headers={"Accept": "application/json"})
                res = request.urlopen(req, context=context)
                # If the API times out due a long running query
                if res.status == 408:
                    # wait just over a minute
                    sleep(61)
                    # then continue this loop with the same URL
                    continue
                elif res.status == 204:
                    # no data so leave loop
                    break
                payload = json.loads(res.read().decode())
                next = payload["next"]
                attempts = 0
            except HTTPError as e:
                if e.code == 408:
                    sleep(61)
                    continue
                else:
                    # If there is a different HTTP error, it wil re-try 3 times before failing
                    if attempts < 3:
                        attempts += 1
                        sleep(61)
                        continue
                    else:
                        sys.stderr.write("LAST URL: " + next)
                        raise e

            for i, item in enumerate(payload["results"]):

                entries = None
                if "entry_subset" in item:
                    entries = item["entry_subset"]
                elif "entries" in item:
                    entries = item["entries"]

                if entries is not None:
                    entries_header = "-".join(
                        [
                            entry["accession"]
                            + "("
                            + ";".join(
                                [
                                    ",".join(
                                        [
                                            str(fragment["start"])
                                            + "..."
                                            + str(fragment["end"])
                                            for fragment in locations["fragments"]
                                        ]
                                    )
                                    for locations in entry["entry_protein_locations"]
                                ]
                            )
                            + ")"
                            for entry in entries
                        ]
                    )
                    f.write(
                        ">"
                        + item["metadata"]["accession"]
                        + HEADER_SEPARATOR
                        + entries_header
                        + HEADER_SEPARATOR
                        + item["metadata"]["name"]
                        + "\n"
                    )
                else:
                    f.write(
                        ">"
                        + item["metadata"]["accession"]
                        + HEADER_SEPARATOR
                        + item["metadata"]["name"]
                        + "\n"
                    )

                seq = item["extra_fields"]["sequence"]
                fastaSeqFragments = [
                    seq[0 + i : LINE_LENGTH + i]
                    for i in range(0, len(seq), LINE_LENGTH)
                ]
                for fastaSeqFragment in fastaSeqFragments:
                    f.write(fastaSeqFragment + "\n")

            # Don't overload the server, give it time before asking for more
            if next:
                sleep(1)


def main(args):
    """
    Downloads the sequences of the Pfam families given in the arguments.
    """

    base_url = "https://www.ebi.ac.uk:443/interpro/api/protein/UniProt/entry/pfam/{}/?page_size=200&extra_fields=sequence"
    os.makedirs("downloads", exist_ok=True)

    for pfam_code in args.pfam_codes:
        if not re.match(r"PF[0-9]{5}", pfam_code):
            raise Exception(f'Pfam code not valid. Must be "PF" followed by 5 digits, got: {pfam_code}. Example: PF12345')
        download_file = os.path.join("downloads", f"{pfam_code}.fasta")
        url = base_url.format(pfam_code)
        logger.info(f"Downloading {pfam_code} from {url}")
        download_pfam_entry(url, download_file)
        logger.info(f"Downloaded {pfam_code} to {download_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pfam_codes",
        nargs="+",
        help='Pfam codes of the families to be downloaded. Must be "PF" and 5 digits. Example: PF12345',
    )
    args = parser.parse_args()
    main(args)
