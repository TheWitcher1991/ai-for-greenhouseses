from sdk.contracts import RegistryCredentials
from sdk.registry.cvat import CvatRegistry

credentials = RegistryCredentials(host="cvat.stgau.ru", login="Ryabokonova.I", password='N$W"Ch|g9R')

cvat_registry = CvatRegistry(credentials)

cvat_registry.save_annotations()
