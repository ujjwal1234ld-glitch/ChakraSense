"""Render a ~122 s landscape reel from the team's original animated MP4.

python submission/make_reel.py ChakraSense_Explainer_v2.mp4 ChakraSense_Reel.mp4
Requires FFmpeg (drawtext, scale, concat, AAC) on PATH or --ffmpeg PATH.
For voice in the results scenes use Windows PowerShell, espeak-ng, or --voices DIR
with 1.wav, 2.wav and 3.wav. No account or network service is used.
"""
import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import wave

NARRATION = [
    "We tested two ways to predict tomorrow's wind. The satellite model uses clouds and wind history. The track model uses pressure, position, and distance to land. A storm stays in just one time period for each experiment.",
    "Across fifty seven later storms, the track model cut average wind error from eleven point seven six to nine point zero five knots. The comparison holds the storm's wind constant for twenty four hours.",
    "Rapid intensification is harder. We detected twelve of fifty windows and raised forty five false flags. The satellite model missed every positive window in twenty fifteen. These are research forecasts for historical replay."
]
SCENES = [
    ("How tomorrow's wind is estimated", "SATELLITE MODEL","Nine images + recent wind", "TRACK MODEL","Pressure + position + wind", "Train on earlier storms. Check the forecasts on later storms."),
    ("How much did wind error fall?", "WIND HELD CONSTANT","11.76 kt", "TRACK MODEL","9.05 kt", "2020-2025 test  |  876 forecasts across 57 storms"),
    ("Which rapid increases were caught?", "RI WINDOWS DETECTED","12 of 50", "FALSE FLAGS","45", "2015 satellite test: 0 of 13 caught. Historical research only.")
]


def first_font():
    candidates=["C:/Windows/Fonts/segoeuib.ttf","/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
    return next((Path(p) for p in candidates if Path(p).exists()),None)


def voice(text, path):
    if platform.system()=="Windows":
        ps=shutil.which("powershell.exe")
        if ps:
            command="Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate=0; $s.SetOutputToWaveFile($env:CS_WAV); $s.Speak($env:CS_TEXT); $s.Dispose()"
            subprocess.run([ps,"-NoProfile","-NonInteractive","-Command",command],
                           env=dict(os.environ,CS_WAV=str(path),CS_TEXT=text),check=True)
            return
    engine=shutil.which("espeak-ng") or shutil.which("espeak")
    if not engine:
        raise RuntimeError("Provide recorded 1.wav, 2.wav and 3.wav with --voices or install espeak-ng.")
    subprocess.run([engine,"-s","155","-w",str(path),text],check=True)


def render(a):
    ff=Path(a.ffmpeg).resolve() if Path(a.ffmpeg).is_file() else shutil.which(a.ffmpeg)
    if not ff:
        raise ValueError("FFmpeg executable unavailable.")
    ff=str(ff)
    source=a.source.resolve()
    dest=a.output.resolve()
    if not source.is_file() or source==dest or dest.exists():
        raise ValueError("Use an existing source and a distinct unused output path.")
    font=Path(a.font) if a.font else first_font()
    if font is None or not font.exists():
        raise ValueError("Pass a TrueType font with --font.")
    if not dest.parent.is_dir():
        dest.parent.mkdir(parents=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as td:
        work=Path(td)
        shutil.copyfile(font,work/"type.ttf")
        def run(args):
            subprocess.run([ff,"-nostdin","-hide_banner","-loglevel","error","-y",*args],cwd=work,check=True)
        enc=subprocess.check_output([ff,"-hide_banner","-encoders"],stderr=subprocess.STDOUT,text=True)
        codecs=[name for name in ("libx264","h264_nvenc","h264_mf","mpeg4") if name in enc]
        for codec in codecs:
            try:
                run(["-f","lavfi","-i","color=s=1280x720:r=30","-t","0.1","-c:v",codec,"probe.mp4"])
                break
            except subprocess.CalledProcessError:
                continue
        else:
            raise ValueError("No working MP4 video encoder found.")
        encoding=["-c:v",codec,"-b:v","2100k","-pix_fmt","yuv420p","-r","30","-c:a","aac",
                  "-b:a","128k","-ar","48000","-ac","1"]
        run(["-i",str(source),"-t","72","-map","0:v:0","-map","0:a:0","-vf","scale=1280:720,setsar=1",
             "-af","afade=t=out:st=71.72:d=0.28",*encoding,"opening.mp4"])
        clips=["opening.mp4"]
        total=72.0
        for i,(title,l1,v1,l2,v2,caption) in enumerate(SCENES,1):
            wav=work/f"{i}.wav"
            if a.voices: shutil.copyfile(a.voices/f"{i}.wav",wav)
            else: voice(NARRATION[i-1],wav)
            with wave.open(str(wav)) as w:
                duration=max(7,w.getnframes()/w.getframerate()+0.6)
            total+=duration
            if total+5.6>149.8:
                raise ValueError("The recorded voices exceed the form limit. Shorten the recordings.")
            # The original animated character corners stay visible behind the new results.
            filt=["scale=1280:720","setsar=1",
                  "drawbox=x=0:y=0:w=1280:h=560:color=0x0c2843:t=fill",
                  "drawbox=x=100:y=550:w=1080:h=170:color=0x0c2843:t=fill",
                  "drawbox=x=145:y=185:w=475:h=290:color=0xf4f9f8:t=fill",
                  "drawbox=x=660:y=185:w=475:h=290:color=0xf4f9f8:t=fill"]
            def txt(content,x,y,size,color,delay=0):
                name=f"s{i}_{len(filt)}.txt"
                (work/name).write_text(content,encoding="utf-8")
                filt.append(f"drawtext=fontfile=type.ttf:textfile={name}:fontsize={size}:fontcolor={color}:x='{x}':y='{y}+12*(1-min(1,max(0,(t-{delay})/0.55)))':alpha='min(1,max(0,(t-{delay})/0.55))'")
            txt("ChakraSense   /   Historical evaluation","(w-tw)/2",46,20,"0x63d8ce")
            txt(title,"(w-tw)/2",98,41,"white",.12)
            for center,label,value,delay in [(382,l1,v1,.4),(898,l2,v2,.65)]:
                txt(label,f"{center}-tw/2",230,22,"0x206d72",delay)
                txt(value,f"{center}-tw/2",304,34 if i==1 else 59,"0x143450",delay+.15)
            txt(caption,"(w-tw)/2",578,24,"white",1.0)
            filt.append("fade=t=in:st=0:d=0.20")
            name=f"scene{i}.mp4"
            run(["-stream_loop","-1","-ss","72","-i",str(source),"-i",str(wav),"-map","0:v:0","-map","1:a:0",
                 "-t",str(duration),"-vf",",".join(filt),"-af","apad",*encoding,name])
            clips.append(name)
        run(["-ss","169.65","-i",str(source),"-t","5.6","-map","0:v:0","-map","0:a:0",
             "-vf","scale=1280:720,setsar=1,fade=t=out:st=5.1:d=0.5","-af","afade=t=in:st=0:d=0.1,afade=t=out:st=5.1:d=0.5",*encoding,"ending.mp4"])
        clips.append("ending.mp4")
        (work/"clips.txt").write_text("".join(f"file '{name}'\n" for name in clips))
        run(["-f","concat","-safe","0","-i","clips.txt","-c","copy","-movflags","+faststart","reel.mp4"])
        run(["-i","reel.mp4","-map","0:v:0","-map","0:a:0","-f","null","-"])
        os.replace(work/"reel.mp4",dest)
        print(f"Rendered {dest.name}. Planned length {total+5.6:.1f} seconds. Check final container duration before submission.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--voices",type=Path)
    p.add_argument("--font")
    p.add_argument("--ffmpeg",default="ffmpeg")
    render(p.parse_args())
