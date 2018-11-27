# Turkey-Turkey

**Data fields**

**vid_id**: YouTube video ID associated with this sample.

**start_time_seconds_youtube_clip**: Where in the YouTube video this audio feature starts.

**end_time_seconds_youtube_clip**: Where in the YouTube video this audio feature ends.

**audio_embedding**: Extracted frame-level audio feature, embedded down to 128 dimensions per frame using AudioSet’s VGGish tools available [here](https://github.com/tensorflow/models/tree/master/research/audioset)

**is_turkey**: The target: whether or not the original audio clip contained a turkey. Label is a soft label, based on whether or not AudioSet’s ontology labeled this clip with “Turkey”, and may count turkey calls and other related content as being “turkey”. is_turkey is 1 if the clip contains a turkey sound, and 0 if it does not.
