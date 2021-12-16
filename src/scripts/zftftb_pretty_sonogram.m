function [IMAGE,F,T]=zftftb_pretty_sonogram(SIGNAL,FS,varargin)
%zftftb_pretty_sonogram computes a simple 2-taper spectrogram using the
%Gauss window and derivative of Gauss (the basis for reassignment).
%
%	SIGNAL
%	vector with microphone data (double)
%
%	FS
%	sampling rate (default: 48e3)
%
%	the following may be passed as parameter/value pairs:
%
%		tscale
%		time scale for Gaussian window for the Gabor transform (in ms, default: 1.5)
%
%		len
%		fft window length (in ms, default: 34)
%
%		nfft
%		number of points in fft (in ms, default: 34)
%
%		overlap
%		window overlap (in ms, default: 33)
%
%		filtering
%		high-pass audio signals (corner Fs in Hz, default: 500)
%
%		norm_amp
%		normalize microphone amplitude to 1 (default: 1)
%
%		zeropad
%		zeropadding of the signal ([] for none, 0 to set to len/2, >0 for zeropad,
%		default: [])
%
%		postproc
%		enable post-processing of spectrogram image ('y' or 'n', default: 'y')
%
%		clipping
%		postproc only, clip the log-amplitude of the spectrogram at this value (default: -2)
%
%		saturation
%		postproc only, image saturation (0-1, default: .8)
%
%

nparams=length(varargin);

if nargin<2 | isempty(FS)
	disp('Setting FS to default: 48e3');
	FS=48e3;
end

overlap=67; % window overlap (ms)
tscale=2; % gauss timescale (ms)
len=70; % window length (ms)
postproc='y'; % postprocessing
nfft=[];
zeropad=[];
norm_amp=0; % normalize amplitude?
filtering=[];
clipping=[-2 2]; % clipping
saturation=.8; % image saturation (0-1)
units='ln'; % units for clipping (ln for natural log, db for db)

if mod(nparams,2)>0
	error('Parameters must be specified as parameter/value pairs!');
end

for i=1:2:nparams
	switch lower(varargin{i})
		case 'overlap'
			overlap=varargin{i+1};
		case 'len'
			len=varargin{i+1};
		case 'tscale'
			tscale=varargin{i+1};
		case 'postproc'
			postproc=varargin{i+1};
		case 'nfft'
			nfft=varargin{i+1};
		case 'low'
			low=varargin{i+1};
		case 'high'
			high=varargin{i+1};
    case 'zeropad'
			zeropad=varargin{i+1};
		case 'norm_amp'
			norm_amp=varargin{i+1};
		case 'filtering'
			filtering=varargin{i+1};
		case 'clipping'
			clipping=varargin{i+1};
		case 'saturation'
			saturation=varargin{i+1};
		case 'units'
			units=varargin{i+1};
		otherwise
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isa(SIGNAL,'double')
	SIGNAL=double(SIGNAL);
end

len=round((len/1e3)*FS);
overlap=round((overlap/1e3)*FS);

if length(SIGNAL)<=len
	warning('Length of signal shorter than window length, trunacting len to %g',num2str(floor(length(SIGNAL)/5)));
	fprintf(1,'\n\n');
	difference=len-overlap;
	len=floor(length(SIGNAL)/3);
	overlap=len-difference;
	nfft=[];
end

if isempty(nfft)
	nfft=2^nextpow2(len);
else
	nfft=2^nextpow2(nfft);
end

if zeropad==0

	% TODO: more accurate zero padding for end of signal (to get true zero phase spectrogram)

	zeropad=round(len/2);
	autopad=1;
else
	autopad=0;
end

if ~isempty(zeropad)
	SIGNAL=[zeros(zeropad,1);SIGNAL(:);zeros(zeropad,1)];
	%disp(['Zero pad: ' num2str(zeropad/FS) ' S']);
end

if norm_amp
	%disp('Normalizing signal amplitude');
	SIGNAL=SIGNAL./max(abs(SIGNAL));
end

if ~isempty(filtering)
	[b,a]=ellip(5,.2,40,[filtering]/(FS/2),'high');
	SIGNAL=filtfilt(b,a,SIGNAL);
end

t=-len/2+1:len/2;
sigma=(tscale/1e3)*FS;
w = exp(-(t/sigma).^2);
dw = -2*w.*(t/(sigma^2));

% take the two spectrograms, use simple "multi-taper" approach

[S,F,T]=spectrogram(SIGNAL,w,overlap,nfft,FS);
[S2]=spectrogram(SIGNAL,dw,overlap,nfft,FS);

% convert to user-designated units

switch lower(units)
	case 'ln'
		IMAGE=log((abs(S)+abs(S2))/2);
	case 'db'
		IMAGE=20*log10((abs(S)+abs(S2))/2);
	otherwise
		IMAGE=(abs(S)+abs(S2))/2;
end

% postproc is generally useful for writing out to an image directly

% slightly elaborate way of clipping and normalizing,
% leaving intact for now to retain legacy compatibility

if lower(postproc(1))=='y'

	if length(clipping)==1
		clipping=[clipping max(IMAGE(:))];
	end

	% clip, map from [0,1]

	IMAGE=min(IMAGE,clipping(2));
	IMAGE=max(IMAGE,clipping(1));
	IMAGE=(IMAGE-clipping(1));
	IMAGE=IMAGE./max(IMAGE(:)); % scale from 0 to 1
	IMAGE=IMAGE*saturation;

end

% if auto zeropad, shift time vector (otherwise let the user do it)

if autopad==1
    %disp('Adjusting time-axis to account for zero pad');
    T=T-zeropad/FS;
end
