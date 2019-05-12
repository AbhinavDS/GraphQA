#TODO
5. Create a Mapping Config for question_parse (when joining ques_parse/gofai)
6. Functional Program learning
7. Check if network params need to be stored


## Sometime
22. think about meta_data file structure
32. debug mac
33. debug evaluator accuracy variation



## IMPORTANT
28. Scenegraph gen:: Find other model
31. stacked attention - AG
36. Non - RL Mertics optimization (2 (AG) + 1 (ADS)): Can explore MultiLabelSoftMarginLoss. Kind of does one versus all. Does not seem to give a major benefit over directly using BCELoss.

37. RL based metric optimization (2 (AG) + 1 (ADS))
38. GOFAI

40. ?? When to add some probability from scenegraph generated!!
41. Test on a data split that is homogenous in structural types: The questions in current splits can be answered with a relatively high confidence if you know the words in the image and the question. Moreover, the mean and standard deviation of number of relate questions is a bit higher in this split than overall dataset which might be the reason behind a higher performance than MAC baseline.
42. Discuss the dimensionality of the Question Encoder Layer in SAN
43. Different AvgPool Layer size for Image and Object Features? Resolve the confusion.
44. Clean up scratch to free up space.
45. Perhaps take some mean for valid and plaus metrics; also check if weights are appropriate
## PERMISSION DENIED SOLVE IT (IMMM)