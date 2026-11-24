# ChakraSense — start here

You now have a working demonstration, a small trained practice model, real cyclone wind records, and the source code for the larger CNN–GRU model.

**The real cyclone prediction model is not trained yet.** It still needs authorised satellite images and a working PyTorch installation. No model can be promised to have almost no faults. This starter makes the remaining gaps visible and checks several common mistakes automatically.

## 1. See it working now — no installation needed

1. If you downloaded the ZIP, right-click it and choose **Extract All**.
2. Open the extracted `ChakraSense` folder.
3. Double-click **Open_Demo.cmd**, or open **ChakraSense_Demo.html** in Edge or Chrome.
4. Choose an artificial storm. Move the time slider.
5. Watch the pictures, estimated wind and RI probability change.

Everything in this demonstration comes from saved model results. The pictures are artificial practice data, not INSAT observations. The model used here is a small image-statistics model, not the CNN–GRU. No live weather is being received.

![How the project works](assets/build_flow.svg)

## 2. Understand the three outputs

- **Current wind:** the model's estimate of the wind at the decision time, measured in knots. One knot is about 1.85 km/h.
- **RI probability:** the estimated chance of wind increasing by at least 30 knots over the next 24 hours. We use IMD/NEWDELHI three-minute winds consistently.
- **Review flag:** a prompt for a person to inspect the result. It is not an official warning.

The displayed wind range is calculated from earlier model errors. Its coverage must be measured on new storms. Even in the practice test, the nominal 90% range covered only 83.3% of held-out examples. That is one reason we do not claim certainty.

## 3. What has already been done

| Part | What you have |
|---|---|
| Practice model | Trained lightweight model and saved predictions |
| Visual demonstration | Offline page with images, time slider, wind chart and comparisons |
| Real cyclone labels | 4,124 usable NEWDELHI records for 144 storms, 2013–2025 |
| Training examples | 432 artificial sequences across 48 whole artificial storms |
| Larger neural model | CNN–GRU source, training, calibration and evaluation commands |
| Data protection against mistakes | Exact 24-hour labels; storm separation; arrival-time checks; file fingerprints |
| Satellite reader | Configurable HDF5 reader and geographic crop code; awaits actual-file verification |
| Local alert test | CAP 1.2 Test/Private XML with a named reviewer; no public transmission |
| Verification | 27 tests passed; 2 dependency-dependent tests skipped |

## 4. Set up Windows for training

Your machine reports an **NVIDIA GeForce RTX 5060 Laptop GPU with about 8 GB of memory**. That is a reasonable starting point for this small model. Actual GPU training could not be verified in this session.

1. Install **Python 3.12** from the official [Python Windows downloads](https://www.python.org/downloads/windows/) if `py -3.12 --version` does not work. You can keep your existing Python 3.14 installation. The tested practice environment uses Python 3.12.
2. Keep this project in a normal local folder with free storage. Satellite downloads can be large; inspect one file's size before requesting many dates.
3. Double-click **Setup_Windows.cmd**. It creates a separate environment for this project and downloads the required libraries. The GPU package alone is several GB.
4. Double-click **Run_Tests.cmd**. Read the last lines. The neural and satellite tests should run, not be skipped, before you rely on those parts.
5. Double-click **Train_Neural_Demo.cmd**. This trains the CNN–GRU using artificial data and saves its result in `artifacts/neural_demo`.

**What blocked training here:** Windows Application Control rejected the newly installed PyTorch `shm.dll` (WinError 4551). Newly installed NumPy also had a blocked DLL. The practice model was run with libraries already supplied in the trusted application runtime. No CNN–GRU weights were produced.

Try the provided setup from your normal Windows session. If the same policy error appears, ask the device administrator to review and approve the official Python/PyTorch packages in an approved environment. Do not turn off Windows protection to make the project run. Reinstalling repeatedly may not solve a policy restriction. CPU mode also needs PyTorch to load, so it does not solve this DLL error.

If setup works but the GPU runs out of memory, open PowerShell in the project folder and reduce the training batch size from 16 to 4. The command is in `README.md`. Close games and other GPU-heavy programs first. GPU compatibility advice is on the [official PyTorch installation page](https://docs.pytorch.org/get-started/locally/).

## 5. Create your MOSDAC account — you must do this part

1. Open [MOSDAC registration](https://mosdac.gov.in/signup/).
2. Enter your own accurate details. Complete email verification and any approval process shown there. Personally review the data-use terms.
3. Sign in and search the INSAT catalogue. Begin with **one satellite and one product family**.
4. Choose an imager product containing thermal infrared and water-vapour observations, temperature conversion information, and matching location coordinates. A coloured preview picture is not a usable training file.
5. Download one or two small sample selections first. Record the satellite, product name, dataset ID, observation times, and file format.
6. Keep your password on your computer. Do not put it in this project's shared ZIP or send it in chat.

For batch downloading, use the official [MOSDAC download tool and manual](https://mosdac.gov.in/downloadapi-manual). It explains how to find the dataset ID and run `mdapi.py`. The starter includes `config/mosdac_config_TEMPLATE.json` as a guide. The official tool expects its local file to be named `config.json`. Keep that credential-bearing copy in a private download folder. Search results can be viewed without authentication; downloading requires your account. Check the current limits shown by MOSDAC before large requests.

The file `data/real/download_plan.csv` lists real storm IDs, names, dates and dataset splits. It is a planning list, not a promise that every satellite file exists. First verify a complete small storm sequence before downloading the whole list.

## 6. Check the first real satellite file

This is the step where help from someone familiar with satellite data is most valuable.

1. Run the `inspect-h5` command in `README.md`. It lists the arrays and their attributes inside your file.
2. Compare these with that product's official manual.
3. Copy `config/insat_mapping_TEMPLATE.json` to `config/insat_mapping.json`.
4. Enter the correct infrared array, water-vapour array, temperature lookup tables, latitude/longitude arrays, scales and missing-value codes.
5. Check that temperatures are in **kelvin**, and locations are in **degrees** after conversion. Raw pixel counts are not temperatures.
6. Check that each channel uses its own correct geographic grid. Different INSAT satellites/channels can have different resolutions.
7. Only after checking, set `verified_against_product_manual` to `true`.

The program deliberately stops while this switch is false. I cannot honestly supply a verified mapping without seeing an actual downloaded file. The current reader supports matching two-dimensional location arrays; products using another projection or coordinate structure need an additional adapter.

## 7. Prepare real examples, then train

1. Fill a copy of `config/raw_manifest_template.csv`. It is a list of your files and their times and locations. The field-by-field explanation is in `DATA_GUIDE.md`.
2. Run **prepare** to make two-channel crops covering the same physical area.
3. Run **build** to assemble nine past image pairs for each decision time. Missing histories are rejected and counted; they are not invented.
4. Read `build_report.json`. Check the number of retained storms and positive RI examples in every split.
5. Train the lightweight model first. Then train the neural model. The exact commands are in `README.md`.
6. Use training storms to learn, validation storms to choose settings, and calibration storms to adjust probabilities and error ranges.
7. Evaluate the final chosen model once on the untouched test storms. Keep this result even if it is disappointing. Do not keep changing the model to make this same test score improve.

The initial real-data periods are: training **2013–2018**, validation **2019**, calibration **2020**, test **2021–2025**. Whole storms stay together. These periods were chosen from label availability before training any real-image model. They replace the unsuitable idea of using 2022 alone for validation, because it has no qualifying positive examples in this imported data.

These are still small samples: only two storms with RI in calibration and six in the planned test period, before missing satellite images are removed. Thousands of overlapping windows do not equal thousands of independent storms. Get expert statistical advice before making strong performance claims.

## 8. What I cannot finish without your help

| What remains | Why it remains | What you should do |
|---|---|---|
| MOSDAC registration and approval | Requires your identity and acceptance of terms | Complete step 5 yourself |
| Authorised INSAT downloads | No account or files supplied | Download sample files, then follow the storm plan |
| Verified satellite mapping | Actual product structure has not been inspected | Check the first HDF5 file and manual with step 6 |
| CNN–GRU execution | Windows blocked its library | Complete approved Windows setup and run tests/demo training |
| Real model training | Real image sequences are missing | Complete steps 6–7 |
| Evidence of real forecast skill | Artificial examples cannot establish it | Run held-out real-storm evaluation and independent review |
| Operational replay | Historical best tracks arrive after the event | Obtain timestamped operational centers and image arrival logs |
| Environment inputs | No suitable timed feed integrated | Add verified forecast-time GFS inputs later; do not substitute delayed reanalysis into a live claim |
| Authorised public alerts | Needs an official operating partner | Work with the responsible weather/disaster authority after validation |

## 9. Your next practical milestone

**Open the demo, get Windows setup passing, and download one real INSAT file containing the two required channels.** After that, the main task is verifying the file mapping and completing a small real storm sequence. You do not need to start with a huge download.

For detailed commands, open `README.md`. For what was actually tested, open `IMPLEMENTATION_STATUS.md`. For the earlier illustrated design explanation, use the separate beginner solution PDF supplied with this project.
