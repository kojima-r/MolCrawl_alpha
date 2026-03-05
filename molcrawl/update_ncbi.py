import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import requests


class NCBIBacteriaUpdaterWebAPI:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.output_dir = Path("assets/genome_species_list/species")
        self.email = "kazunobu.matsubara@riken.jp"  # NCBI recommended: Email address settings

    def search_bacteria_genomes(self, retmax=5000):
        """Search bacterial genome ID using NCBI E-utils"""
        print("🔍 Searching bacteria genomes via NCBI E-utils...")

        # Search query: complete bacterial genome or chromosome level
        search_url = f"{self.base_url}/esearch.fcgi"
        params = {
            "db": "genome",
            "term": 'bacteria[organism] AND ("complete genome"[Properties] OR "chromosome"[Properties])',
            "retmax": retmax,
            "retmode": "json",
            "email": self.email,
        }

        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()

            if "esearchresult" in data and "idlist" in data["esearchresult"]:
                genome_ids = data["esearchresult"]["idlist"]
                print(f"✅ Found {len(genome_ids)} genome records")
                return genome_ids
            else:
                print("❌ No genome IDs found")
                return []

        except Exception as e:
            print(f"❌ Error searching genomes: {e}")
            return []

    def fetch_genome_details(self, genome_ids, batch_size=100):
        """Get detailed information from genome ID"""
        print(f"📋 Fetching details for {len(genome_ids)} genomes...")

        all_organisms = set()

        for i in range(0, len(genome_ids), batch_size):
            batch_ids = genome_ids[i : i + batch_size]
            print(f"  Processing batch {i // batch_size + 1}/{(len(genome_ids) - 1) // batch_size + 1}")

            fetch_url = f"{self.base_url}/efetch.fcgi"
            params = {
                "db": "genome",
                "id": ",".join(batch_ids),
                "retmode": "xml",
                "email": self.email,
            }

            try:
                response = requests.get(fetch_url, params=params)
                response.raise_for_status()

                # parse XML
                organisms = self.parse_genome_xml(response.text)
                all_organisms.update(organisms)

                # Wait for a while due to NCBI API limitations
                time.sleep(0.5)

            except Exception as e:
                print(f"  ⚠️ Error in batch {i // batch_size + 1}: {e}")
                continue

        print(f"✅ Extracted {len(all_organisms)} unique organisms")
        return sorted(list(all_organisms))

    def parse_genome_xml(self, xml_content):
        """Extract organism name from XML"""
        organisms = set()

        try:
            root = ET.fromstring(xml_content)

            # Find organism name (multiple possible paths)
            for genome in root.findall(".//Genome"):
                # Find the OrganismName element
                org_name = genome.find(".//OrganismName")
                if org_name is not None and org_name.text:
                    cleaned = self.clean_organism_name(org_name.text)
                    if cleaned:
                        organisms.add(cleaned)

                # Alternative path: ProjectName etc.
                proj_name = genome.find(".//ProjectName")
                if proj_name is not None and proj_name.text:
                    cleaned = self.clean_organism_name(proj_name.text)
                    if cleaned:
                        organisms.add(cleaned)

        except ET.ParseError as e:
            print(f"  ⚠️ XML parsing error: {e}")

        return organisms

    def clean_organism_name(self, name):
        """Cleaning and standardizing organism names to genus level"""
        if not name:
            return None

        name = name.strip()

        # remove unnecessary information
        for pattern in [
            " strain ",
            " substrain ",
            " subsp.",
            " var.",
            " serovar ",
            " biovar ",
        ]:
            if pattern in name:
                name = name.split(pattern)[0]

        # remove inside parentheses
        if "(" in name:
            name = name.split("(")[0].strip()

        # Remove numbers and code parts
        parts = []
        for part in name.split():
            # Skip parts that are obviously not genus names
            if len(part) < 3:
                continue
            if part.lower() in ["sp.", "bacterium", "strain", "isolate"]:
                continue
            if part.replace("-", "").replace("_", "").replace(".", "").isdigit():
                continue
            parts.append(part)

        if not parts:
            return None

        # Special handling of Candidatus
        if len(parts) >= 2 and parts[0].lower() == "candidatus":
            return f"Candidatus {parts[1]}"

        # Usually returns only the genus name
        return parts[0]

    def get_bacteria_via_taxonomy(self):
        """Obtain bacterial classification from NCBI Taxonomy"""
        print("🌳 Fetching bacteria taxonomy...")

        # Get subclassification from bacterial taxon ID (2)
        search_url = f"{self.base_url}/esearch.fcgi"
        params = {
            "db": "taxonomy",
            "term": "bacteria[subtree] AND genus[rank]",
            "retmax": 10000,
            "retmode": "json",
            "email": self.email,
        }

        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()

            if "esearchresult" in data and "idlist" in data["esearchresult"]:
                tax_ids = data["esearchresult"]["idlist"]
                print(f"✅ Found {len(tax_ids)} bacterial genera")

                # get classification name
                genera = self.fetch_taxonomy_names(tax_ids)
                return genera
            else:
                return []

        except Exception as e:
            print(f"❌ Error fetching taxonomy: {e}")
            return []

    def fetch_taxonomy_names(self, tax_ids, batch_size=200):
        """Get the classification name from the classification ID"""
        print("📋 Fetching taxonomy names...")

        genera = set()

        for i in range(0, len(tax_ids), batch_size):
            batch_ids = tax_ids[i : i + batch_size]

            fetch_url = f"{self.base_url}/efetch.fcgi"
            params = {
                "db": "taxonomy",
                "id": ",".join(batch_ids),
                "retmode": "xml",
                "email": self.email,
            }

            try:
                response = requests.get(fetch_url, params=params)
                response.raise_for_status()

                # Extract genus name from XML
                names = self.parse_taxonomy_xml(response.text)
                genera.update(names)

                time.sleep(0.3)  # Compatible with API restrictions

            except Exception as e:
                print(f"  ⚠️ Error in taxonomy batch: {e}")
                continue

        return sorted(list(genera))

    def parse_taxonomy_xml(self, xml_content):
        """Extract genus name from Taxonomy XML"""
        genera = set()

        try:
            root = ET.fromstring(xml_content)

            for taxon in root.findall(".//Taxon"):
                # get ScientificName
                sci_name = taxon.find(".//ScientificName")
                if sci_name is not None and sci_name.text:
                    name = sci_name.text.strip()
                    # Genus level name only (no spaces)
                    if " " not in name and len(name) > 2:
                        genera.add(name)

        except ET.ParseError:
            pass

        return genera

    def update_bacteria_list(self):
        """Main update process"""
        print("🧬 NCBI Bacteria List Updater (Web API)")
        print("=" * 50)

        # Method 1: Obtain from genome database
        genome_ids = self.search_bacteria_genomes(retmax=3000)  # The latest 3000 items
        organisms_from_genome = []
        if genome_ids:
            organisms_from_genome = self.fetch_genome_details(genome_ids)

        # Method 2: Get from classification database
        organisms_from_taxonomy = self.get_bacteria_via_taxonomy()

        # merge both results
        all_organisms = set(organisms_from_genome + organisms_from_taxonomy)
        final_list = sorted(list(all_organisms))

        print("\n📊 Results:")
        print(f"  From genomes: {len(organisms_from_genome)}")
        print(f"  From taxonomy: {len(organisms_from_taxonomy)}")
        print(f"  Combined unique: {len(final_list)}")

        # Compare with existing listand save
        self.save_updated_list(final_list)

        return True

    def save_updated_list(self, new_organisms):
        """Save updated list"""
        output_file = self.output_dir / "bacteria.txt"

        # Compare with existing list
        current_organisms = set()
        if output_file.exists():
            with open(output_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("//"):
                        current_organisms.add(line)

        new_set = set(new_organisms)
        added = new_set - current_organisms
        removed = current_organisms - new_set

        print("\n📈 Changes:")
        print(f"  Current: {len(current_organisms)}")
        print(f"  New: {len(new_set)}")
        print(f"  Added: {len(added)}")
        print(f"  Removed: {len(removed)}")

        # backup
        if output_file.exists():
            backup_file = output_file.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            backup_file.write_text(output_file.read_text())
            print(f"💾 Backup: {backup_file}")

        # save new list
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(f"// filepath: {output_file}\n")
            f.write(f"// Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"// Total genera: {len(new_organisms)}\n")
            for organism in new_organisms:
                f.write(f"{organism}\n")

        print(f"✅ Updated list saved: {output_file}")

        # Detailed report if there are any changes
        if added or removed:
            self.save_change_report(added, removed)


def main():
    updater = NCBIBacteriaUpdaterWebAPI()
    success = updater.update_bacteria_list()

    if success:
        print("\n🎉 Update completed successfully!")
    else:
        print("\n❌ Update failed")


if __name__ == "__main__":
    main()
