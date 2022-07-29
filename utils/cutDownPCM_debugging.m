Fs = 16e3;
range = 1:(Fs*60);
mic = pcm2wav('mic', Fs);
wav2pcm('mic', mic(range));
ref = pcm2wav('ref', Fs);
wav2pcm('ref', ref(range));
target = pcm2wav('target', Fs);
wav2pcm('target', target(range));