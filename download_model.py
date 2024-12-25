from google.colab import files
import shutil

pasta_para_zipar = '.ctesibioAI-model'
file_zip = 'ctesibioAI-model.zip' 


shutil.make_archive(base_name=file_zip.replace('.zip', ''), format='zip', root_dir=pasta_para_zipar)

files.download(file_zip)