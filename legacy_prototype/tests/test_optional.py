"""Run these in an approved Windows environment after installing requirements."""
import unittest,tempfile,json
from pathlib import Path
import numpy as np

class NeuralModel(unittest.TestCase):
    def test_cnn_gru_forward_backward(self):
        try:
            import torch
            from chakrasense.model import CycloneNet
        except (ImportError,OSError) as e:self.skipTest('PyTorch unavailable: '+str(e)[:160])
        m=CycloneNet();x=torch.randn(2,9,2,64,64);y=m(x)
        self.assertEqual(tuple(y['wind'].shape),(2,));(y['wind'].square().mean()+y['logit'].square().mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

class SatelliteAdapter(unittest.TestCase):
    def fixture(self,path):
        import h5py
        lat,lon=np.meshgrid(np.linspace(11.5,12.5,64),np.linspace(77.5,78.5,64),indexing='ij')
        with h5py.File(path,'w') as f:
            f['lat']=lat;f['lon']=lon;f['IR']=np.full((64,64),250.);f['WV']=np.full((64,64),240.)
        channel=lambda name:{'image':name,'units':'kelvin','image_fill_values':[],'latitude':'lat','longitude':'lon','latitude_scale':1,'latitude_offset':0,'longitude_scale':1,'longitude_offset':0,'coordinate_fill_values':[]}
        return {'verified_against_product_manual':True,'satellite':'FIXTURE','pixels':32,'width_km':80,'max_neighbour_km':3,'channels':{'ir':channel('IR'),'wv':channel('WV')}}
    def test_same_geographic_crop(self):
        try:from chakrasense.satellite import crop_h5
        except (ImportError,OSError) as e:self.skipTest('HDF5/scipy unavailable: '+str(e)[:160])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'fixture.h5';mapping=self.fixture(path);a=crop_h5(path,mapping,12,78)
            self.assertEqual(a.shape,(2,32,32));self.assertTrue((a[0]==250).all());self.assertTrue((a[1]==240).all())
            mapping['verified_against_product_manual']=False
            with self.assertRaises(ValueError):crop_h5(path,mapping,12,78)
