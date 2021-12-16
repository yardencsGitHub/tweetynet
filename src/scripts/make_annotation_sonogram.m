% this script makes a spectrogram for the annotation figure using the matlab pretty_sonogram script
function make_annotation_sonogram(path_to_wav_file,tlim,flim)
    clipping = [-2 2];
    disp_band=cell2mat(flim); %[500 1e4];
    time_limits = cell2mat(tlim); %[3.09 4.1];
    [y,fs] = audioread(path_to_wav_file);
    [s,f,t]=zftftb_pretty_sonogram(y,fs,'len',10.3,'overlap',8.3,...
                'zeropad',0,'norm_amp',1,'clipping',clipping,'nfft',1024);
    %[s,f,t,P] = spectrogram((y/(sqrt(mean(y.^2)))),440*2,2*(440-88),1024*1,fs);
    s = log(abs(s)+1);
    startidx=max([find(f<=disp_band(1))]); stopidx=min([find(f>=disp_band(2))]);
    mint_idx = max([find(t<=time_limits(1))]); maxt_idx = min([find(t>=time_limits(2))]);
    im=s(startidx:stopidx,mint_idx:maxt_idx)*64;
    im=flipdim(im,1);
    imwrite(uint8(63-im),colormap([ 'gray' '(63)']),[ path_to_wav_file '.gif'],'gif');

    [s,f,t,P] = spectrogram((y/(sqrt(mean(y.^2)))),440*0.75,(440-88)*0.75,1024*1,fs);
    fh = figure('Visible','off'); imagesc(t,f,log(abs(s)+1));
    axis xy
    xlim(time_limits)
    ylim(disp_band)
    colormap(1-gray)
    caxis([0.25 6])
    xticks([]); yticks([]);

    saveas(fh,[path_to_wav_file '.png'])


    