#conversor de .soil a .txt para visualizar en ipynb "visualizar_soil.ipynb"
def convertir_soil_a_txt(input_path, output_path):
    """
    Convierte un archivo .soil (formato CABO) a un archivo .txt plano, línea por línea.
    No interpreta el contenido, simplemente lo transfiere tal cual.
    """
    with open(input_path, 'r') as infile:
        contenido = infile.read()

    with open(output_path, 'w') as outfile:
        outfile.write(contenido)

    print(f"Archivo convertido: {output_path}")
    
convertir_soil_a_txt("data/soil/ec2.soil", "data/soil/ec2_convertido.txt")
