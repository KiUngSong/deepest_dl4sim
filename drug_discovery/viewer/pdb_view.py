import py3Dmol


# Function to display a protein structure from a PDB ID
def show_pdb_from_file(pdb_file):
    with open(pdb_file, "r") as file:
        pdb_data = file.read()
    view = py3Dmol.view(width=400, height=400)
    view.addModel(pdb_data, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    view.show()
