"""Build the academic reel from the supplied 1280x720 animated explainer.
Usage: python make_reel.py original.mp4 ChakraSense_Reel.mp4 --ffmpeg /path/to/ffmpeg
Requires Pillow, FFmpeg and Windows built-in English speech, or four prerecorded
PCM WAV files in --voices. Optional --music takes a licensed instrumental bed.
"""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave

from PIL import Image, ImageDraw, ImageFont

RATE=30
VOICE=[
    "Our prototype now tests two models. One studies the satellite pictures and past winds. The other follows the storm's wind, pressure, and distance from land.",
    "Here is a real forecast from May sixteenth, twenty twenty. The wind started at forty knots. The track model predicted fifty nine. The next day, the observed wind was seventy five.",
    "Over eight hundred and seventy six forecasts from fifty seven later storms, the average wind error fell from eleven point seven six knots to nine point zero five. That is a twenty three percent improvement over holding wind constant.",
    "Rapid strengthening is harder. The track model caught twelve of fifty cases, missed thirty eight, and raised forty five false flags. The satellite candidate missed all thirteen cases in its twenty fifteen test. These are research results."
]
CAPS=[
    [("Two research models forecast wind 24 hours ahead.","Each also scores rapid intensification.")],
    [("A real case: 16 May 2020 at 15:00 UTC.","Wind at issue was 40 knots."),("The forecast was 58.8 knots.","The storm reached 75 knots the next day.")],
    [("We tested 876 forecasts across 57 later storms.","Test period: 2020 to 2025."),("Wind error fell from 11.76 to 9.05 knots.","Holding wind constant is the comparison.")],
    [("The track model caught 12 of 50 RI windows.","It missed 38 and raised 45 false flags."),("The satellite candidate missed all 13 in 2015.","These are historical research results.")]
]
PS=r"""
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null=[Windows.Media.SpeechSynthesis.SpeechSynthesizer,Windows.Media.SpeechSynthesis,ContentType=WindowsRuntime]
$null=[Windows.Media.SpeechSynthesis.SpeechSynthesisStream,Windows.Media.SpeechSynthesis,ContentType=WindowsRuntime]
$s=New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$v=[Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object DisplayName -Like '*Zira*' | Select-Object -First 1
if($v){$s.Voice=$v}
$op=$s.SynthesizeTextToStreamAsync($env:CS_TEXT)
$m=[System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {$_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'} | Select-Object -First 1
$task=$m.MakeGenericMethod([Windows.Media.SpeechSynthesis.SpeechSynthesisStream]).Invoke($null,@($op))
$task.Wait()
$r=[System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($task.Result)
$o=[System.IO.File]::Create($env:CS_WAV)
try{$r.CopyTo($o)}finally{$o.Dispose();$r.Dispose();$s.Dispose()}
"""

def narrate(text,path):
    subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-Command",PS],check=True,
                   env=dict(os.environ,CS_TEXT=text,CS_WAV=str(path)))


def ease(t):
    t=min(1,max(0,t))
    return t*t*(3-2*t)


def artwork(scene,t,duration,fonts,sprites):
    im=Image.new("RGB",(1280,720),"#102e49")
    d=ImageDraw.Draw(im)
    pale="#f4f9fa";navy="#173344";teal="#00756e";mint="#5ce4d4";gold="#ffd276"
    def txt(x,y,s,size=24,fill="white",anchor="la",bold=False):
        d.text((x,y),s,font=fonts[(size,bold)],fill=fill,anchor=anchor)
    # A slow-moving field and the original character sprites keep the new scenes in motion.
    for j in range(54):
        x=(j*163+int(t*(3+j%4)))%1280
        y=64+(j*83)%480
        d.ellipse((x,y,x+2,y+2),fill="#29506a")
    d.rounded_rectangle((62,38,70,74),radius=4,fill=mint)
    txt(88,38,"ChakraSense",26,bold=True)
    txt(1218,47,"HISTORICAL REPLAY",18,mint,"ra",True)
    titles=["Two ways to read a storm","What happened the next day?","Wind error across 57 storms","Which rapid increases were caught?"]
    txt(640,96+int(12*(1-ease(t/.6))),titles[scene],41,"white","ma",True)
    def card(x,y,w,h):
        d.rounded_rectangle((x+2,y+7,x+w+2,y+h+7),radius=19,fill="#0a233b")
        d.rounded_rectangle((x,y,x+w,y+h),radius=19,fill=pale)
    if scene==0:
        for i,(lab,tit,sub) in enumerate([("01   SATELLITE","Cloud structure","Nine images + recent wind"),("02   WIND AND TRACK","Storm history","Wind, pressure + coast proximity")]):
            x=114+i*536; y=207+int(30*(1-ease((t-.25*i)/.85)))
            card(x,y,506,249);txt(x+30,y+25,lab,19,teal,bold=True)
            for j in range(7):
                h=int((20+6*j+10*math.sin(j+t))*ease((t-.15*j)/.75))
                d.rounded_rectangle((x+32+j*30,y+139-h,x+49+j*30,y+139),radius=3,fill=teal if i==0 else "#d09227")
            txt(x+28,y+151,tit,33,navy,bold=True);txt(x+28,y+203,sub,22,"#3e6271")
    elif scene==1:
        card(118,190,1044,288)
        for i,(lab,value,col) in enumerate([("Wind at issue",40,"#70929c"),("24-hour forecast",58.8,teal),("Observed after 24 h",75,"#c78326")]):
            y=239+i*78; e=ease((t-i*1.65)/1.15)
            txt(147,y,lab,23,navy,bold=True);d.rounded_rectangle((395,y+2,395+max(1,int(650*value/86*e)),y+33),radius=8,fill=col)
            if e>.07:txt(406+int(650*value/86*e),y+1,format(value,"g")+" kt",22,navy,bold=True)
    elif scene==2:
        for i,(lab,val,col) in enumerate([("WIND HELD CONSTANT",11.76,"#577688"),("WIND AND TRACK MODEL",9.0464,teal)]):
            x=114+i*535;card(x,195,507,277);txt(x+28,222,lab,20,teal,bold=True)
            grow=ease((t-i*.6)/1.3)
            txt(x+27,270,f"{val*grow:.2f}",69,navy,bold=True);txt(x+30,366,"knots of mean absolute error",21,"#42606d")
            d.rounded_rectangle((x+29,419,x+30+int(410*val/12*grow),438),radius=8,fill=col)
        txt(640,497,"23.1% lower average wind error",29,mint,"ma",True)
    else:
        card(118,185,616,313);txt(147,207,"50 observed RI windows",27,navy,bold=True)
        for j in range(50):
            x=151+(j%10)*54;y=277+(j//10)*38;s=ease((t-.04*j)/.4)
            r=int(12*s)
            d.ellipse((x-r,y-r,x+r,y+r),fill=teal if j<12 else "#cfdbdd")
        card(761,185,400,313)
        for i,(n,l,col) in enumerate([("12","detected",teal),("38","missed","#566b77"),("45","false flags","#a86c1e")]):
            y=212+85*i;txt(795,y,n,45,col,bold=True);txt(885,y+15,l,25,navy,bold=True)
    # These sprites are crops of successive frames from the user's animation, not new characters.
    pair=sprites[int(t*8)%len(sprites)]
    im.paste(pair[0],(0,553));im.paste(pair[1],(1125,553))
    d.rounded_rectangle((164,567,1116,683),radius=17,fill="#0a2032")
    cap=CAPS[scene][min(len(CAPS[scene])-1,int(t/(duration+.001)*len(CAPS[scene])))]
    txt(640,586,cap[0],24,"white","ma",True);txt(640,631,cap[1],21,gold,"ma")
    return im


def main(args):
    ff=shutil.which(args.ffmpeg) or str(Path(args.ffmpeg).resolve())
    source=args.source.resolve();dest=args.output.resolve()
    if not source.is_file() or source==dest or dest.exists():
        raise ValueError("Use an existing source and a separate unused output path")
    dest.parent.mkdir(parents=True,exist_ok=True)
    def run(xs):
        return subprocess.run([ff,"-nostdin","-hide_banner","-loglevel","error","-y",*map(str,xs)],check=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as folder:
        tmp=Path(folder)
        wavs=[];dur=[]
        for i,s in enumerate(VOICE):
            wav=tmp/f"voice{i}.wav"
            if args.voices:shutil.copyfile(args.voices/f"{i+1}.wav",wav)
            else:narrate(s,wav)
            with wave.open(str(wav)) as w:seconds=w.getnframes()/w.getframerate()
            wavs.append(wav);dur.append(math.ceil((seconds+.55)*RATE)/RATE)
        if not 90<=72+sum(dur)+5.2<=149.8:raise ValueError(f"Reel length out of bounds with voice segments {dur}")
        encoders=subprocess.check_output([ff,"-hide_banner","-encoders"],stderr=subprocess.STDOUT,text=True)
        for codec in ("libx264","h264_nvenc","h264_mf","mpeg4"):
            if codec not in encoders:continue
            try:
                run(["-f","lavfi","-i","color=s=1280x720:r=30","-t","0.1","-c:v",codec,tmp/"probe.mp4"]);break
            except subprocess.CalledProcessError:continue
        else:raise RuntimeError("No working video encoder")
        encode=["-c:v",codec,"-b:v","1800k","-pix_fmt","yuv420p","-r",str(RATE),"-c:a","aac","-b:a","128k","-ac","1","-ar","48000"]
        run(["-i",source,"-t","72","-map","0:v:0","-map","0:a:0","-vf","setsar=1","-af","afade=t=out:st=71.75:d=.24".replace("d=.24","d=0.24"),*encode,tmp/"open.mp4"])
        run(["-ss","72","-i",source,"-t","3","-vf","fps=8","-q:v","3",tmp/"sprite%02d.jpg"])
        sprites=[]
        for p in sorted(tmp.glob("sprite*.jpg")):
            with Image.open(p) as im:
                sprites.append((im.crop((0,553,145,720)),im.crop((1125,553,1280,720))))
        regular=args.font or Path("C:/Windows/Fonts/segoeui.ttf")
        bold=args.bold or Path("C:/Windows/Fonts/segoeuib.ttf")
        fonts={(sz,b):ImageFont.truetype(str(bold if b else regular),sz) for sz in (18,19,20,21,22,23,24,25,26,27,29,33,41,45,69) for b in (True,False)}
        clips=[tmp/"open.mp4"]
        for i,(wav,dt) in enumerate(zip(wavs,dur)):
            out=tmp/f"result{i}.mp4"
            au=["-i",str(wav)]
            if args.music:
                au+=["-stream_loop","-1","-ss",str(i*11),"-i",str(args.music.resolve()),"-filter_complex","[1:a]apad[a];[2:a]volume=0.12[b];[a][b]amix=inputs=2:duration=first:normalize=0[vo]","-map","0:v:0","-map","[vo]"]
            else:au+=["-af","apad","-map","0:v:0","-map","1:a"]
            cmd=[ff,"-nostdin","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","1280x720","-r",str(RATE),"-i","pipe:0",*au,"-t",str(dt),*encode,str(out)]
            with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
                for j in range(round(dt*RATE)):proc.stdin.write(artwork(i,j/RATE,dt,fonts,sprites).tobytes())
                proc.stdin.close()
                if proc.wait()!=0:raise RuntimeError("Motion graphics encoding failed")
            clips.append(out)
        run(["-ss","169.65","-i",source,"-t","5.2","-map","0:v:0","-map","0:a:0","-vf","setsar=1,fade=t=out:st=4.7:d=0.5","-af","afade=t=in:st=0:d=0.12,afade=t=out:st=4.7:d=0.5",*encode,tmp/"close.mp4"])
        clips.append(tmp/"close.mp4");lst=tmp/"edit.txt"
        lst.write_text("".join(f"file '{p.name}'\n" for p in clips))
        run(["-f","concat","-safe","0","-i",lst,"-c","copy","-movflags","+faststart",tmp/"final.mp4"])
        run(["-xerror","-i",tmp/"final.mp4","-map","0:v","-map","0:a","-f","null","-"])
        os.replace(tmp/"final.mp4",dest)
        print(f"Rendered {dest.name}. Segment durations {dur}. Approximate total {77.2+sum(dur):.2f} s.")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--ffmpeg",default="ffmpeg");p.add_argument("--font",type=Path);p.add_argument("--bold",type=Path)
    p.add_argument("--voices",type=Path);p.add_argument("--music",type=Path)
    main(p.parse_args())
