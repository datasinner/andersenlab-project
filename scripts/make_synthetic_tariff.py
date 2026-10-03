"""Render the synthetic test tariff: an invented port with a different structure.

    uv run python scripts/make_synthetic_tariff.py   # writes data/synthetic/...pdf

The generalisation test (BUILD_PLAN §13): this document differs from the TNPA
tariff book in layout (portrait, one column, unnumbered headings, comma
thousands separators, euros) and in mechanics (dues on net tonnage, a per-day
charge beyond a free period, a minimum, mutually exclusive reductions plus a
stackable one, pilotage by length bands, towage that joins a tug-allocation
table to a charge-per-tugs table, marginal length tiers given as widths, a
levy per gross ton with a maximum, an exemption). The system must price it
with no code change. Expected values are worked out by hand in
eval/cases/exampleville_nordic_tern.json.
"""

from pathlib import Path

import pymupdf

OUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "synthetic"
    / "exampleville_port_charges_2025.pdf"
)

CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.35; }
h1 { font-size: 18pt; font-weight: bold; margin: 0 0 4pt 0; }
h2 { font-size: 13pt; font-weight: bold; margin: 14pt 0 4pt 0; }
p { margin: 0 0 5pt 0; }
table { border-collapse: collapse; margin: 4pt 0 8pt 0; }
th, td { border: 1px solid black; padding: 3pt 8pt; font-size: 10pt; }
th { font-weight: bold; }
"""

HTML = """
<h1>Exampleville Harbour Board</h1>
<p><b>Schedule of Port Charges 2025</b></p>
<p>This schedule applies to vessels calling at the Port of Exampleville from 1 January 2025 to
31 December 2025. All charges are in euros (EUR) and exclude VAT at 21%.</p>

<h2>Definitions</h2>
<p>“Gross tonnage” (GT) and “net tonnage” (NT) mean the tonnages stated in the vessel’s
International Tonnage Certificate (1969).</p>
<p>“Length overall” (LOA) means the vessel’s length overall in metres.</p>
<p>“Operation” means one movement of a vessel: its arrival, its departure, or a shift between
berths.</p>
<p>“Day” means a period of 24 hours, or part of it, that the vessel spends at a berth.</p>

<h2>Harbour Dues</h2>
<p>Harbour dues are payable by every vessel entering the port, once per call.</p>
<p>Per 100 net tonnage or part thereof .......... 4.20</p>
<p>For each day or part thereof beyond the fifth day at berth, per 100 net tonnage or part
thereof .......... 0.60</p>
<p>Minimum per call .......... 250.00</p>
<p><b>Reductions</b></p>
<p>• Liner vessels calling at the port at least 12 times a year: 15%.</p>
<p>• Vessels calling only to take on bunkers or stores and staying less than 24 hours: 30%.</p>
<p>These two reductions cannot be combined.</p>
<p>• Vessels with an Environmental Ship Index (ESI) score of 30 or more: a further 10%, in
addition to any other reduction.</p>
<p>Vessels of the national navy and the Harbour Board’s own craft are exempt from harbour
dues.</p>

<h2>Pilotage</h2>
<p>Pilotage is compulsory for every vessel of 70 metres LOA or more. It is charged per operation
according to the vessel’s length overall:</p>
<table>
<tr><th>Length overall</th><th>Charge per operation (EUR)</th></tr>
<tr><td>Up to 100 m</td><td>1,200.00</td></tr>
<tr><td>100.01 m to 200 m</td><td>2,450.00</td></tr>
<tr><td>Over 200 m</td><td>3,900.00</td></tr>
</table>
<p>A surcharge of 25% applies to operations that start or end between 22:00 and 06:00.</p>

<h2>Towage</h2>
<p>Towage is compulsory for vessels over 120 metres LOA. The harbour master allocates tugs
according to the vessel’s gross tonnage, and towage is charged per operation according to the
number of tugs allocated.</p>
<table>
<tr><th>Gross tonnage</th><th>Tugs allocated</th></tr>
<tr><td>Up to 10,000</td><td>1</td></tr>
<tr><td>10,001 to 35,000</td><td>2</td></tr>
<tr><td>Over 35,000</td><td>3</td></tr>
</table>
<table>
<tr><th>Tugs allocated</th><th>Charge per operation (EUR)</th></tr>
<tr><td>1</td><td>1,650.00</td></tr>
<tr><td>2</td><td>3,150.00</td></tr>
<tr><td>3</td><td>4,600.00</td></tr>
</table>
<p>An additional tug ordered by the master is charged at 1,650.00 per operation.</p>

<h2>Berth Dues</h2>
<p>Berth dues are charged for each day or part thereof at berth, per metre of length overall or
part thereof:</p>
<table>
<tr><th>Length</th><th>EUR per metre per day</th></tr>
<tr><td>First 100 metres</td><td>1.10</td></tr>
<tr><td>Next 100 metres</td><td>0.80</td></tr>
<tr><td>Each metre beyond 200 metres</td><td>0.50</td></tr>
</table>

<h2>Conservancy Levy</h2>
<p>A conservancy levy of 0.035 per gross ton is payable once per call, up to a maximum of
2,500.00 per call.</p>

<h2>Waste Reception Fee</h2>
<p>Every vessel pays a waste reception fee of 180.00 per call, whether or not it delivers waste.
Vessels holding a valid waste delivery exemption certificate issued by the Harbour Board are
exempt.</p>

<h2>Passenger Levy</h2>
<p>Passenger vessels pay 2.50 for each passenger embarking or disembarking.</p>

<h2>Yacht Dues</h2>
<p>Pleasure craft pay 1.80 per metre of length overall for each day or part thereof in port.</p>

<h2>Services on Request</h2>
<p>Fresh water: 3.20 per tonne supplied. Shore power: 0.28 per kWh. Additional garbage collection:
quoted on request.</p>

<h2>Wharfage</h2>
<p>Wharfage on cargo loaded or discharged is 0.90 per tonne and is payable by the cargo owner.</p>

<h2>Licences</h2>
<p>Ship agent licence: 400.00 per year. Stevedoring licence: 1,200.00 per year.</p>
"""


def render(path: Path = OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    story = pymupdf.Story(html=HTML, user_css=CSS)
    page = pymupdf.paper_rect("a4")
    area = page + (56, 56, -56, -56)
    writer = pymupdf.DocumentWriter(str(path))
    more = True
    while more:
        device = writer.begin_page(page)
        more, _ = story.place(area)
        story.draw(device)
        writer.end_page()
    writer.close()
    return path


if __name__ == "__main__":
    print(f"wrote {render()}")
