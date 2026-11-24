import argparse,json,platform,sys,urllib.request
from pathlib import Path
from .data import import_ibtracs,read_tracks,create_splits,save_json,build_sequences

NOAA_URL='https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv'

def main():
    p=argparse.ArgumentParser(description='ChakraSense research prototype. Start with START_HERE.md.');s=p.add_subparsers(dest='command',required=True)
    s.add_parser('doctor')
    a=s.add_parser('download-labels');a.add_argument('--raw',default='data/raw/ibtracs.NI.list.v04r01.csv');a.add_argument('--out',default='data/real/tracks.csv');a.add_argument('--start',type=int,default=2013);a.add_argument('--end',type=int,default=2025)
    a=s.add_parser('import-labels');a.add_argument('raw');a.add_argument('--out',default='data/real/tracks.csv');a.add_argument('--start',type=int,default=2013);a.add_argument('--end',type=int,default=2025)
    a=s.add_parser('split');a.add_argument('tracks');a.add_argument('--out',required=True);a.add_argument('--validation-year',type=int,default=2019);a.add_argument('--calibration-year',type=int,default=2020);a.add_argument('--test-year',type=int,default=2021)
    a=s.add_parser('inspect-h5');a.add_argument('file');a.add_argument('--out')
    a=s.add_parser('prepare');a.add_argument('raw_manifest');a.add_argument('mapping');a.add_argument('--out',required=True)
    a=s.add_parser('build');a.add_argument('frames');a.add_argument('tracks');a.add_argument('splits');a.add_argument('--out',required=True);a.add_argument('--mode',choices=['retrospective','operational'],default='retrospective');a.add_argument('--steps',type=int,default=9);a.add_argument('--hours',type=int,default=3)
    a=s.add_parser('demo-data');a.add_argument('--out',default='data/demo')
    a=s.add_parser('train');a.add_argument('manifest');a.add_argument('--out',required=True);a.add_argument('--epochs',type=int,default=8);a.add_argument('--batch-size',type=int,default=16);a.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=s.add_parser('evaluate');a.add_argument('manifest');a.add_argument('run')
    a=s.add_parser('predict');a.add_argument('run');a.add_argument('sequence')
    a=s.add_parser('train-baseline');a.add_argument('manifest');a.add_argument('--out',required=True)
    a=s.add_parser('evaluate-baseline');a.add_argument('manifest');a.add_argument('run')
    a=s.add_parser('predict-baseline');a.add_argument('run');a.add_argument('sequence')
    a=s.add_parser('dashboard');a.add_argument('manifest');a.add_argument('run');a.add_argument('--out',default='ChakraSense_Demo.html')
    a=s.add_parser('test-cap');a.add_argument('--storm',required=True);a.add_argument('--probability',required=True,type=float);a.add_argument('--reviewer',required=True);a.add_argument('--out',required=True)
    args=p.parse_args()
    if args.command=='doctor':
        import torch
        result={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    elif args.command in ['download-labels','import-labels']:
        if args.command=='download-labels':
            dest=Path(args.raw);dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                temp=dest.with_suffix('.partial');urllib.request.urlretrieve(NOAA_URL,temp);temp.replace(dest)
        result=import_ibtracs(args.raw,args.out,args.start,args.end)
    elif args.command=='split':
        result=create_splits(read_tracks(args.tracks),args.validation_year,args.calibration_year,args.test_year);save_json(args.out,result)
    elif args.command=='inspect-h5':
        from .satellite import inspect_h5
        result=inspect_h5(args.file)
        if args.out:save_json(args.out,result)
    elif args.command=='prepare':
        from .satellite import prepare_frames
        result=prepare_frames(args.raw_manifest,args.mapping,args.out)
    elif args.command=='build':
        result={'sequences':len(build_sequences(args.frames,args.tracks,args.splits,args.out,args.mode,args.steps,args.hours))}
    elif args.command=='demo-data':
        from .demo import generate_demo
        result={'manifest':generate_demo(args.out)}
    elif args.command=='train':
        from .training import train
        result=train(args.manifest,args.out,args.epochs,args.batch_size,args.device)
    elif args.command=='evaluate':
        from .training import evaluate
        result=evaluate(args.manifest,args.run)
    elif args.command=='predict':
        from .training import predict_crop
        result=predict_crop(args.run,args.sequence)
    elif args.command=='train-baseline':
        from .baseline import train_baseline
        result=train_baseline(args.manifest,args.out)
    elif args.command=='evaluate-baseline':
        from .baseline import evaluate_baseline
        result=evaluate_baseline(args.manifest,args.run)
    elif args.command=='predict-baseline':
        from .baseline import predict_baseline
        result=predict_baseline(args.run,args.sequence)
    elif args.command=='dashboard':
        from .dashboard import export_dashboard
        result=export_dashboard(args.manifest,args.run,args.out)
    elif args.command=='test-cap':
        from .alerts import create_test_cap
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        create_test_cap(args.storm,args.probability,args.reviewer,args.out);result={'local_test_file':args.out,'public_delivery':False}
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
