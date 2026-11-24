# Milestone form responses

Questions 1–6 were outside the supplied screenshots. Use the official registration details for those fields.

**Q7 · Project title**  
ChakraSense: 24-Hour Cyclone Wind and Rapid Intensification Forecasting

**Q8 · Deliverable type**  
App with code

**Q9 · Project design (391/500 characters)**  
ChakraSense has two retrospective forecast paths. One combines nine infrared and water-vapour frames with past wind. The other uses wind history, pressure, position and distance to land. Both estimate wind 24 hours ahead and score rapid intensification. Storm-level and chronological splits separate training from evaluation. The design diagrams show data flow, models and historical replay.

**Q10 · Project design upload**  
Upload `ChakraSense_Design.pdf`.

**Q11 · Running implementation**  
The Python prototype reconstructs historical storm windows, checks the timestamps and future wind labels, trains regression and RI models, and evaluates forecasts on storms from later years. Satellite processing uses NumPy and pandas. The models use scikit-learn. The wind-and-track test covered 876 forecasts across 57 storms in 2020–2025. Average wind error was 9.05 kt, compared with 11.76 kt for holding wind constant. The model detected 12 of 50 RI windows and raised 45 false flags. In the 2015 satellite test, both tested RI models missed all 13 positive windows. Live ingestion and prospective validation remain future work.

**Q12 · Complete project folder**  
https://github.com/ujjwal1234ld-glitch/ChakraSense/tree/milestone-forecast-evaluation

**Q13 · Reel**  
Paste the anyone-can-view Drive or unlisted YouTube link and the verified length in seconds. The original 177.8-second explainer exceeds the 150-second limit.

**Q14 · Supporting document**  
Technical Report.

**Q15 · PDF upload**  
Use `ChakraSense_TechnicalReport.pdf`. Replace the project-name prefix with your registered team name before uploading if it differs. Leave the registered name blank until confirmed.

**Q16 · Prototype status (two options)**  
Some modules working. For the second choice, “more than 60% completed” is a suggested estimate for the retrospective prototype: data preparation, training, forecasting and evaluation work. The form provides no official percentage rubric. Live integration and prospective validation are future work.

**Q17 · Original Milestone One PDF**  

**Q18 · Actual member contributions**  

**Q19 · Challenges (249 characters)**  
Rapid intensification is rare. On 57 later storms the wind-and-track model cut average wind error by 23%, but it caught only 12 of 50 intensifying windows and raised 45 false flags. Historical best-track data also limits claims about live forecasts.

**Q20 · Team declaration**  
Select only after every member has reviewed and agreed to the contribution record.
