from google.colab import files
import shutil

# Substitua 'nome_da_pasta' pelo caminho da pasta que você quer baixar
pasta_para_zipar = '.gpt2-finetuned'
file_zip = 'ctesibioAI-model.zip'


shutil.make_archive(base_name=file_zip.replace('.zip', ''), format='zip', root_dir=pasta_para_zipar)

files.download(file_zip)