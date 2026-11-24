import tempfile,unittest,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from chakrasense.data import make_labels,create_splits,build_sequences,validate_samples,read_tracks,save_json,dataset_fingerprint
from chakrasense.evaluation import metrics,fit_calibration,calibrate,conformal_radius,average_precision_score,logistic_fit
from chakrasense.baseline import fit,infer,predict_baseline
from chakrasense.alerts import create_test_cap,NS

class Labels(unittest.TestCase):
    def test_exact_24h_threshold(self):
        x=pd.DataFrame({'storm_id':['A']*3,'time':['2020-01-01','2020-01-02','2020-01-03'],'wind_kt':[20,50,79]})
        y=make_labels(x);self.assertEqual(y.ri.tolist(),[1,0]);self.assertEqual(len(y),2)
    def test_missing_future_is_not_negative(self):
        x=pd.DataFrame({'storm_id':['A','A'],'time':['2020-01-01','2020-01-02T03:00Z'],'wind_kt':[20,80]})
        self.assertTrue(make_labels(x).empty)
    def test_storm_identity_prevents_cross_pairing(self):
        x=pd.DataFrame({'storm_id':['A','B'],'time':['2020-01-01','2020-01-02'],'wind_kt':[20,80]})
        self.assertTrue(make_labels(x).empty)
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):make_labels(pd.DataFrame({'storm_id':['A','A'],'time':['2020-01-01']*2,'wind_kt':[20,30]}))
    def test_cross_year_storm_kept_together(self):
        x=pd.DataFrame({'storm_id':['A','A','B','C','D'],'time':['2018-12-31','2019-01-01','2019-06-01','2020-06-01','2021-06-01']})
        self.assertEqual(create_splits(x,2019,2020,2021),{'A':'train','B':'validation','C':'calibration','D':'test'})
    def test_agencies_cannot_be_mixed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tracks.csv';pd.DataFrame({'storm_id':['A'],'time':['2020-01-01'],'wind_kt':[30],'agency':['USA'],'wind_averaging_minutes':[1]}).to_csv(p,index=False)
            with self.assertRaises(ValueError):read_tracks(p)

class SequenceTiming(unittest.TestCase):
    def fixture(self,root,late=False,known=True,center='operational_bulletin'):
        root=Path(root)
        for i in range(2):np.savez(root/f'{i}.npz',bt=np.ones((2,32,32),np.float32)*(240+i),width_km=1000.)
        pd.DataFrame({'storm_id':['A']*2,'observed_at':['2020-01-01T00:00Z','2020-01-01T02:30Z'],
            'available_at':['2020-01-01T00:10Z','2020-01-01T04:00Z' if late else '2020-01-01T02:50Z'],
            'availability_known':[known]*2,'center_availability_known':[known]*2,'center_available_at':['2020-01-01T00:00Z']*2,
            'center_source':[center]*2,'frame_path':['0.npz','1.npz'],'kind':['real']*2,'satellite':['fixture']*2}).to_csv(root/'frames.csv',index=False)
        pd.DataFrame({'storm_id':['A']*2,'time':['2020-01-01T03:00Z','2020-01-02T03:00Z'],'wind_kt':[30,65],
            'agency':['NEWDELHI']*2,'wind_averaging_minutes':[3]*2,'kind':['real']*2}).to_csv(root/'tracks.csv',index=False)
        save_json(root/'splits.json',{'A':'train'})
    def build(self,d,mode='operational'):
        d=Path(d);return build_sequences(d/'frames.csv',d/'tracks.csv',d/'splits.json',d/'out',mode,2,3)
    def test_timely_observations_pass(self):
        with tempfile.TemporaryDirectory() as d:self.fixture(d);self.assertEqual(len(self.build(d)),1)
    def test_late_arrival_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(d,late=True)
            with self.assertRaises(ValueError):self.build(d)
    def test_unknown_arrival_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(d,known=False)
            with self.assertRaises(ValueError):self.build(d)
    def test_best_track_center_rejected_operationally(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(d,center='best_track')
            with self.assertRaises(ValueError):self.build(d)
    def test_retrospective_allows_best_track_but_labels_mode(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(d,center='best_track');self.assertEqual(self.build(d,'retrospective')['mode'].iloc[0],'retrospective')
    def test_invalid_temperatures_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(d);np.savez(Path(d)/'0.npz',bt=np.zeros((2,32,32)),width_km=1000.)
            with self.assertRaises(ValueError):self.build(d)

class Evaluation(unittest.TestCase):
    def test_undefined_far_is_null(self):
        m=metrics([20,20],[20,20],[0,0],[0,1],.5,.5);self.assertIsNone(m['false_alarm_ratio']);self.assertEqual(m['detection'],0)
    def test_average_precision_handles_ties(self):
        self.assertAlmostEqual(average_precision_score([0,1],[.5,.5]),.5)
        self.assertAlmostEqual(average_precision_score([0,1],[.1,.9]),1.)
    def test_brier_climatology_zero_skill(self):
        self.assertEqual(metrics([1,1],[1,1],[.5,.5],[0,1],.5,.5)['brier_skill'],0.)
    def test_calibration_requires_two_classes(self):
        with self.assertRaises(ValueError):fit_calibration([1,2],[1,1])
    def test_calibration_finite_for_extreme_logits(self):
        p=calibrate([-1e9,1e9],{'slope':1.,'intercept':0.});self.assertTrue(np.isfinite(p).all());self.assertTrue(((p>=0)&(p<=1)).all())
    def test_small_interval_calibration_rejected(self):
        with self.assertRaises(ValueError):conformal_radius([1,2])
    def test_finite_sample_interval_rank(self):self.assertEqual(conformal_radius(np.arange(1,11)),10.)
    def test_baseline_learns_and_uses_saved_transform(self):
        x=np.arange(40,dtype=float).reshape(20,2);m=fit(x,x[:,0]*2+10,(x[:,0]>19).astype(int));w,l=infer(m,x)
        self.assertLess(np.mean(abs(w-(x[:,0]*2+10))),10.);self.assertLess(l[0],l[-1])

class Integrity(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);rows=[]
        for i,s in enumerate(['train','validation','calibration','test']):
            np.savez(root/f'{i}.npz',bt=np.full((2,2,32,32),230+i,np.float32),width_km=1000.)
            rows.append({'sample_id':str(i),'storm_id':str(i),'time':f'{2018+i}-01-01','path':f'{i}.npz','split':s,'wind_kt':30,'ri':i%2,'kind':'real','mode':'retrospective','steps':2,'hours':3,'pixels':32,'width_km':1000.})
        p=root/'samples.csv';pd.DataFrame(rows).to_csv(p,index=False);return p
    def test_storm_leak_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.fixture(d);x=pd.read_csv(p);x['storm_id']='same';x.to_csv(p,index=False)
            with self.assertRaises(ValueError):validate_samples(p)
    def test_duplicate_pixel_content_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.fixture(d);np.savez(Path(d)/'1.npz',bt=np.full((2,2,32,32),230,np.float32),width_km=1000.)
            with self.assertRaises(ValueError):validate_samples(p)
    def test_dataset_change_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.fixture(d);old=dataset_fingerprint(p);np.savez(Path(d)/'1.npz',bt=np.full((2,2,32,32),235,np.float32),width_km=1000.)
            self.assertNotEqual(old,dataset_fingerprint(p))
    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.fixture(d);x=pd.read_csv(p);x.loc[0,'path']='../escape.npz';x.to_csv(p,index=False)
            with self.assertRaises(ValueError):validate_samples(p)
    def test_demo_prediction_matches_saved_test_result(self):
        base=Path(__file__).resolve().parents[1];run=base/'artifacts/demo_baseline'
        if not (run/'test_predictions.csv').exists():self.skipTest('Run baseline demo first.')
        r=pd.read_csv(run/'test_predictions.csv').iloc[0];ans=predict_baseline(run,base/'data/demo/sequences'/r.path)
        self.assertAlmostEqual(ans['wind_kt'],r.wind_pred,places=8);self.assertAlmostEqual(ans['ri_probability'],r.probability,places=8)

class LocalAlerts(unittest.TestCase):
    def test_reviewer_required(self):
        with self.assertRaises(ValueError):create_test_cap('test',.5,'','unused.xml')
    def test_cap_test_status_and_schema(self):
        try:from lxml import etree
        except ImportError:self.skipTest('Install lxml for official schema validation.')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'alert.xml';create_test_cap('ARTIFICIAL',.5,'Automated software test',p)
            schema=etree.XMLSchema(etree.parse(str(Path(__file__).parent/'fixtures/CAP-v1.2.xsd')));schema.assertValid(etree.parse(str(p)))
            root=etree.parse(str(p));self.assertEqual(root.findtext('{'+NS+'}status'),'Test');self.assertEqual(root.findtext('{'+NS+'}scope'),'Private')

if __name__=='__main__':unittest.main()
