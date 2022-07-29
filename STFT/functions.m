% *******************************************************
% delay-matching between two signals (complex/real-valued)
% Markus Nentwig, 2005-2012
%
% signal and ref are two sampled signals, same number of points.
% both are treated as cyclic (use zero-padding, if needed)
% 
% * output:
%   => coeff: 
%      complex scaling factor that scales 'ref' into 'signal'
%   => delay 'deltaN' samples (with subsample resolution)
%      applying deltaN delay to 'ref' turns it into 'signal'
%   => 'shiftedRef': 'ref' with 'coeff' and 'deltaN' applied 
%      For example, norm(signal - shiftedRef) should be at a minimum.
% 
% Properly used, the timing can be resolved with an accuracy of a small 
% fraction of the sample duration, the "PART I" algorithm may reach 
% nano-sample accuracy for noise-free signals.
%
% The most recent version can be found here: 
% http://www.dsprelated.com/showcode/207.php
%
% description of the least-squares based algorithm ('PART I'):
% http://www.dsprelated.com/showarticle/26.php
%
% description of the iterative algorithm ('PART II'):
% http://www.dsprelated.com/showcode/288.php
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
% *******************************************************

% ######### "PART 1" ########
% The algorithm as advertised above.
% Call the function from outside this script.
function [coeff, shiftedRef, deltaN] = fitSignal_120825(signal, ref, opts)
    if nargin < 2
        error('need two arguments');
    elseif nargin == 2
        opts = struct();
    end
    
    def = struct();

    % force use of an alternative iterative algorithm
    % less accurate at high signal-to-noise ratio, slower, but more robust
    % to noisy signals and heavy group delay variations
    def.forceIterativeAlgorithm = false;
    
    % standard ('PART I') algorithm
    % high accuracy
    % fast 
    % fails for too noisy signals
    % fails for heavy group delay distortion
    def.enableCoarseCorr = true;
    
    % phase unwrapping
    % can be set (together with enableCoarseCorr = false) to
    % effectively use a conventional phase-unwrapping algorithm.
    % fails if the phase "tracking" is lost across regions of 
    % low signal-to-noise ratio, for example 
    % - when a DC term is removed from a measurement
    % - in a multi-channel signal
    % - under narrow-band interference
    % - due to frequency-selective fading in a radio channel
    def.enableUnwrap = false; 
    
    % generate some plots (others remain disabled, edit the code)
    def.doPlot = false;
    
    % checks, whether the crosscorrelation peak with the 
    % shifted and scaled reference signal is at zero, as it
    % should be.
    def.enableSanityCheck = false;

    % *******************************************************
    % merge default options
    % *******************************************************
    tmp = fieldnames(def);
    for ix = 1 : numel(tmp)
        if ~isfield(opts, tmp{ix})
            opts.(tmp{ix}) = def.(tmp{ix});
        end
    end
    
    % *******************************************************
    % use PART II algorithm, if requested. 
    % otherwise, it is a fallback solution only.
    % *******************************************************    
    if opts.forceIterativeAlgorithm
        [coeff, shiftedRef, deltaN] = mn_fitSignal_corrSearch(signal, ref);
        return;
    end    
    
    signal = signal(:) .'; % force row vector
    ref = ref(:) .';
    n = numel(signal);
    assert(n == numel(ref), 'length mismatch');

    % *******************************************************
    % Calculate the normalized frequency for each FFT bin
    % [-0.5..0.5[
    % *******************************************************
    binFreq=(mod(((0:n-1)+floor(n/2)), n)-floor(n/2))/n;
    
    % *******************************************************
    % Convert signals to frequency domain
    % *******************************************************
    sig_FD = fft(signal);
    ref_FD = fft(ref);
    
    % rotate each bin backwards with the phase of the reference
    u = sig_FD .* conj(ref_FD);
    
    negateCoeff = false;
    if opts.enableCoarseCorr
        
        % *******************************************************
        % Coarse correction (+/- 1 sample) using crosscorrelation between 
        % signal and reference...
        % *******************************************************
        NyqBin = [];
        if mod(n, 2) == 0
            % for an even-sized FFT, it is a matter of interpretation, whether the center
            % bin should be treated as a positive or a negative frequency. 
            % Effectively, it describes a series such as [-1, 1, -1, 1, -1, ...] in the time
            % domain. 
            % Interpolation between samples is not possible, as the direction of the rotating
            % phasor is not known. 
            % In a real-valued signal, it is effectively "half positive, half negative frequency".
            % This definition allows ifft(fft([-1, 1, -1, 1, ...]) to return the original signal.
            % still, it cannot be evaluated between samples.
            % 
            % Making a long story short:
            % Disregard it.
            u(n/2 + 1) = 0;
        end
        Xcor = ifft(u);
        
        % *******************************************************
        % Each bin in Xcor corresponds to a delay in samples.
        % The bin with the highest absolute value corresponds to
        % the delay where maximum correlation occurs.
        % *******************************************************
        ix = find(abs(Xcor) == max(abs(Xcor)));
        ix = ix(1); % in case there are several bitwise-identical peaks
        
        % *******************************************************
        % A negative peak implies a sign change: signal matches -ref.
        % *******************************************************
        if real(Xcor(ix)) > 0
            % positive correlation peak
        else
            % negative correlation peak
            % This would result in a 180 degree baseline phase in the 
            % LS solver, which creates ambiguity problems 
            % Solution: Negate the signal (and all dependent expressions), 
            % and undo by negating coeff at the end. 
            signal = -signal; 
            sig_FD = -sig_FD;
            u = -u;  
            Xcor = -Xcor;
            negateCoeff = true;
        end

        % the location of the peak denotes the delay, with a resolution of 
        % one sample. Delay 0 appears in bin 1 etc.
        integerDelay = ix - 1; 
        
        if opts.doPlot
            figure(); hold on; 
            v = abs(Xcor);
            plot(v, 'b');
            h = plot(integerDelay+1, v(integerDelay+1), 'r+');
            set(h, 'lineWidth', 3);
            legend('crosscorrelation', 'coarse (integer) delay estimate');
            title('coarse delay estimation via crosscorrelation');
        end
    else
        integerDelay = 0;
    end
    
    % *******************************************************
    % un-delay the signal by the estimated coarse delay.
    % causality is not an issue, as the signal is cyclic (it
    % repeats continuously over time = -inf..inf)
    % *******************************************************
    % exp(-2i pi f) is the Fourier transform of a unity delay
    % exp(2i pi f) is a negative delay.
    rotN = exp(2i*pi*integerDelay .* binFreq);
    
    if opts.doPlot
        tref = ifft(fft(ref) .* conj(rotN)); % conj => sign change of delay
        figure(); hold on;
        plot(real(ref), 'b');
        plot(real(signal), 'k');
        plot(real(tref), 'r');
        legend('original reference', 'signal', 'reference, coarse aligned');
    end
    
    if false && opts.doPlot
        figure(); hold on;
        plot(fftshift(binFreq), fftshift(20*log10(abs(fft(ref)) + eps)), 'k');
        plot(fftshift(binFreq), fftshift(20*log10(abs(fft(signal)) + eps)), 'b');
        legend('reference', 'signal');
        xlabel('normalized frequency');
        ylabel('dB');
        title('power spectrum');
    end

    % *******************************************************
    % Find the exact delay using linear least mean squares fit
    % on the phase.
    % 
    % u is the phase difference on any frequency bin. 
    % Delay appears as linear increase in phase, but unwrapping
    % the phase (2 pi ambiguity) across gaps in the power spectrum 
    % is not robust enough.
    % Therefore, use the coarse delay (with +/- 1/2 sample accuracy) to 
    % rotate back the phase. This removes most of the 2 pi steps.
    % Note this is method is not completely "foolproof" either but 
    % usually quite robust. 
    % In case of failure, try oversampling (fft;zero pad; ifft) on 
    % both signal and reference.
    % *******************************************************
    angCorr = angle(u .* rotN);
    ang = angCorr;

    if opts.enableUnwrap
        ang = fftPhaseUnwrap(ang);
    end
    if false && opts.doPlot
        angRaw = angle(u);
        figure(); grid on; hold on;
        h = plot(angRaw, 'k'); set(h, 'lineWidth', 2);
        h = plot(angCorr, 'b'); set(h, 'lineWidth', 2);
        h = plot(ang, 'g'); set(h, 'lineWidth', 1);
        xlabel('FFT bin index');
        ylabel('phase / radians');
        title('phase');
        legend('raw phase', 'phase corrected with coarse estimate', 'corrected and unwrapped');
    end
    
    % *******************************************************
    % For the least-squares fit, the phase is weighted according to
    % the product of amplitudes from signal and reference. 
    % As intuitive explanation: any bin is disregarded, if either
    % reference signal or received signal contain too little energy, 
    % as the phase would be meaningless and vary wildly as result of 
    % added noise etc.
    % *******************************************************
    weight = abs(u);
    % normalize for plotting (doesn't affect the result)
    weight = weight / max(weight);
    
    % Phase shift of a complex-valued scaling factor
    % It rotates all frequencies by the same amount.
    % Apply per-frequency weighting. 
    constRotPhase = 1 .* weight;
    
    % Phase shift of a unit delay
    % Apply per-frequency weighting
    uDelayPhase = -2*pi*binFreq .* weight;    
    
    % The observed phase difference between signal and reference
    % (corrected by the coarse estimate), and weighted.
    ang = ang .* weight;
    
    if false
        % discard low-value bins (optimization for long, highly oversampled signals)
        chkThr = abs(ang);
        thr = max(chkThr) * 1e-6;
        ixPow = find(chkThr > thr);
        constRotPhase = constRotPhase(ixPow);
        uDelayPhase = uDelayPhase(ixPow);
        ang = ang(ixPow);
    end
    
    % Least-squares equation system that attempts to "explain" the observed
    % phase in terms of
    % - a constant phase shift => r(1)
    % - a unity delay => r(2)
    % base .' * r = ang .'
    % least-squares solve for r:
    base = [constRotPhase; uDelayPhase];
    r = base .' \ ang.'; %linear mean square solution
    
    if opts.doPlot
        figure(); hold on;
        h = plot(ang, 'k'); set(h, 'lineWidth', 2);
        plot(constRotPhase, 'g');
        plot(uDelayPhase, 'b');
        plot(base.' * r, 'r');
        title('phase vectors in least-squares solver');
        legend('signal, weighted', 'constant, weighted', 'unit delay, weighted', 'least-squares fit');
        xlabel('FFT bin index'); ylabel('weighted phase');
    end
        
    % The constant phase rotation r(1) is not used
    % It will be obtained later as the phase of 'coeff'
    
    % least-squares optimal number of unit delays 
    fractionalDelay = r(2);
    if opts.enableCoarseCorr
        % assert(abs(fractionalDelay) <= 0.5*1.2345, 'mismatch between coarse- and fine estimation'); 
        if abs(fractionalDelay) > 0.5*1.2345
            disp('*** fitSignal: reverting to iterative algorithm ***');
            [coeff, shiftedRef, deltaN] = mn_fitSignal_corrSearch(signal, ref);
            return;
        end
    end
    
    % *******************************************************
    % Final delay estimate: Restore the coarse estimate that
    % had been removed earlier
    % (returned result)
    % *******************************************************
    deltaN = integerDelay + fractionalDelay;        

    % *******************************************************
    % Convert numbers beyond the midpoint into a negative
    % number (optional)
    % *******************************************************
    deltaN = mod(deltaN-n/2, n) + n/2;
    
    % *******************************************************
    % Delay ref with the final delay estimate
    % *******************************************************    
    ref_FD = ref_FD .* exp(-2i*pi*deltaN .* binFreq);
    shiftedRef = ifft(ref_FD);
    
    % *******************************************************
    % Crosscorrelation with the now time-aligned signal
    % The resulting coeff minimizes norm(signal - coeff * shiftedRef)
    % (returned result)
    % *******************************************************
    coeff = sum(signal .* conj(shiftedRef)) / sum(shiftedRef .* conj(shiftedRef));
    if negateCoeff
        % sign was changed earlier. Change it back.
        coeff = -coeff;
    end
    
    % *******************************************************
    % apply the coefficient to the delayed reference signal. 
    % (returned result)
    % *******************************************************
    shiftedRef = shiftedRef * coeff;

    % *******************************************************
    % Sanity check:
    % calculate the crosscorrelation for all delays. 
    % Assuming we have done a proper job, the peak must be 
    % at zero delay. Otherwise, it would mean that there is a 
    % "better" delay on the tested one-sample grid.
    % *******************************************************
    if opts.enableSanityCheck
        u = sig_FD .* conj(ref_FD);
        Xcor=abs(ifft(u));
        if Xcor(1) ~= max(Xcor)
            figure(); plot(abs(Xcor)); title('sanity check failed');
            error ('not locked to the correct bin');
        end
        assert(real(Xcor(1)) >= 0, 'sign is wrong?!');
    end

    % *******************************************************
    % Report a delay beyond the midpoint as negative
    % *******************************************************
    halfLen = floor(n / 2);
    deltaN = mod(deltaN + halfLen, n) - halfLen;                    
end

% p is a phase of a complex-valued FFT, with the "0 Hz bin" at index 1
% removes phase jumps in excess of pi in positive and negative direction
function p = fftPhaseUnwrap(p)
    n = numel(p);
    lastPosIx = ceil(n/2); % last positive frequency
    
    % positive frequencies - search from zero in positive direction
    ixVec = 2:lastPosIx;
    % ixVecAdj points towards the adjacent bin, in direction
    % towards the origin (minus one bin for pos. frequencies)
    ixVecAdj = ixVec - 1;
    delta = p(ixVec) - p(ixVecAdj);
    delta = round(delta / 2 / pi) * 2 * pi;
    delta = cumsum(delta);
    p(ixVec) = p(ixVec) - delta;
    
    % negative frequencies
    % search from zero in negative direction
    % flip by reverse indexing, then use fast cumsum
    ixVec = n:-1:lastPosIx+1;
    % ixVecAdj points towards the adjacent bin, in direction
    % towards the origin (plus one bin for neg. frequencies)
    ixVecAdj = ixVec + 1;
    ixVecAdj(1) = 1; % n+1 wraps around to 1
    delta = p(ixVec) - p(ixVecAdj);
    delta = round(delta / 2 / pi) * 2 * pi;
    delta = cumsum(delta);
    p(ixVec) = p(ixVec) - delta;    
end

% ****************************************************************
% ######### "PART 2" ########
% 
% The functions below implement a fallback solution, if the above
% algorithm fails. This variant performs better at low signal-to-
% noise ration.
% The algorithm iteratively determines the crosscorrelation,
% predicts the peak location between bins, time-shifts and repeats.
% Documentation:
% http://www.dsprelated.com/showcode/288.php
% It is not as accurate and slower as the first algorithm, but should
% always converge, even if "the" delay between two signals is not well-
% defined because of group delay variations, multiple periods etc.
% ****************************************************************

% wrapper function to provide the same interface as the original 
% fitSignal function
function [coeff, shiftedRef, deltaN] = mn_fitSignal_corrSearch(signal, ref)
    % ****************************************************************
    % estimate delay
    % ****************************************************************
    [deltaN, coeff] = iterDelayEst(signal, ref);
    % this subroutine uses a different definition
    coeff = 1 / coeff;
    deltaN = -deltaN;
    
    % ****************************************************************
    % correct it
    % ****************************************************************
    shiftedRef = cs_delay(ref, 1, deltaN);
    shiftedRef = shiftedRef * coeff; 
end

% ****************************************************************
% estimates delay and scaling factor 
% ****************************************************************
function [delay_samples, coeff] = iterDelayEst(s1, s2)
    
    s1 = s1(:) .'; % force row vectors
    s2 = s2(:) .';
    rflag = isreal(s1) && isreal(s2);
    n = numel(s1);
    halfN = floor(n/2);
    assert(numel(s2) == n, 'signals must have same length');

    % ****************************************************************
    % constants
    % ****************************************************************    
    % exit if uncertainty below threshold
    thr_samples = 1e-6;

    % exit after fixed number of iterations
    nIter = 30;

    % frequency domain representation of signals
    fd1 = fft(s1);
    fd2 = fft(s2);    

    % first round: No delay was applied
    tau = [];
    fd2Tau = fd2; % delayed s2 in freq. domain
    
    % frequency corresponding to each FFT bin -0.5..0.5
    f = (mod(((0:n-1)+floor(n/2)), n)-floor(n/2))/n;

    % uncertainty plot data
    e = [];

    % normalization factor
    nf = sqrt((fd1 * fd1') * (fd2 * fd2')) / n; % normalizes to 1
    
    % search window: 
    % known maximum and two surrounding points
    x1 = -1;
    x2 = -1;
    x3 = -1;
    y1 = -1;
    y2 = -1;
    y3 = -1;
    
    % ****************************************************************
    % iteration loop
    % ****************************************************************
    for count = 1:nIter
    
        % ****************************************************************
        % crosscorrelation with time-shifted signal
        % ****************************************************************
        xcorr = abs(ifft(fd2Tau .* conj(fd1)))/ nf;

        % ****************************************************************
        % detect peak
        % ****************************************************************
        if isempty(tau)

            % ****************************************************************
            % startup
            % initialize with three adjacent bins around peak
            % ****************************************************************
            ix = find(xcorr == max(xcorr));
            ix = ix(1); % use any, if multiple bitwise identical peaks
            
            % indices of three bins around peak
            ixLow = mod(ix-1-1, n) + 1; % one below
            ixMid = ix;
            ixHigh = mod(ix-1+1, n) + 1; % one above

            % delay corresponding to the three bins
            tauLow = mod(ixLow -1 + halfN, n) - halfN;
            tauMid = mod(ixMid -1 + halfN, n) - halfN;         
            tauHigh = mod(ixHigh -1 + halfN, n) - halfN; 

            % crosscorrelation value for the three bins
            xcLow = xcorr(ixLow);
            xcMid = xcorr(ixMid);
            xcHigh = xcorr(ixHigh);
            
            x1 = tauLow;
            x2 = tauMid;
            x3 = tauHigh;
            y1 = xcLow;
            y2 = xcMid; 
            y3 = xcHigh;
        else
            % ****************************************************************
            % only main peak at first bin is of interest
            % ****************************************************************
            tauMid = tau;
            xcMid = xcorr(1);

            if xcMid > y2
                % ****************************************************************
                % improve midpoint
                % ****************************************************************
                if tauMid > x2
                    % midpoint becomes lower point
                    x1 = x2;
                    y1 = y2;
                else
                    % midpoint becomes upper point
                    x3 = x2;
                    y3 = y2;
                end
                x2 = tauMid;
                y2 = xcMid;
            
            elseif tauMid < x2
                % ****************************************************************
                % improve low point
                % ****************************************************************
                assert(tauMid >= x1); % bitwise identical is OK
                assert(tauMid > x1 || xcMid > y1); % expect improvement
                x1 = tauMid;
                y1 = xcMid;
            elseif tauMid > x2 
                % ****************************************************************
                % improve high point
                % ****************************************************************
                assert(tauMid <= x3); % bitwise identical is OK                
                assert((tauMid < x3) || (xcMid > y3)); % expect improvement
                x3 = tauMid;
                y3 = xcMid;
            else
                assert(false, '?? evaluated for existing tau ??');
            end
        end

        % ****************************************************************
        % calculate uncertainty (window width)
        % ****************************************************************
        eIter = abs(x3 - x1);
        e = [e, eIter];
        if eIter < thr_samples
            % disp('threshold reached, exiting');
            break;
        end

        if y1 == y2 || y2 == y3
            % reached limit of numerical accuracy on one side
            usePoly = 0;
        else
            % ****************************************************************
            % fit 2nd order polynomial and find maximum
            % ****************************************************************
            num = (x2^2-x1^2)*y3+(x1^2-x3^2)*y2+(x3^2-x2^2)*y1;
            denom = (2*x2-2*x1)*y3+(2*x1-2*x3)*y2+(2*x3-2*x2)*y1;
            if denom ~= 0
                tau = num / denom;
                % is the point within [x1, x3]? 
                usePoly = ((tau > x1) && (tau < x3));
            else
                usePoly = 0;
            end            
        end
        if ~usePoly
            % revert to linear interpolation on the side with the
            % less-accurate outer sample 
            % Note: There is no guarantee that the side with the more accurate
            % outer sample is the right one, as the samples aren't 
            % placed on a regular grid!
            % Therefore, iterate to improve the "worse" side, which will
            % eventually become the "better side", and iteration converges.
            
            tauLow = (x1 + x2) / 2;
            tauHigh = (x2 + x3) / 2;
            if y1 < y3
                o = [tauLow, tauHigh];
            else
                o = [tauHigh, tauLow];                
            end
            % don't choose point that is identical to one that is already known
            tau = o(1);
            if tau == x1 || tau == x2 || tau == x3
                tau = o(2);
                if tau == x1 || tau == x2 || tau == x3
                    break;
                end
            end
        end

        % ****************************************************************
        % advance 2nd signal according to location of maximum
        % phase shift in frequency domain - delay in time domain
        % ****************************************************************
        fd2Tau = fd2 .* exp(2i * pi * f * tau);
    end % for

    % ****************************************************************
    % plot the uncertainty (window size) over the number of iterations
    % ****************************************************************
    if false
        figure(); semilogy(e, '+-'); grid on;
        xlabel('iteration');
        title('uncertainty in delay');
    end

    % ****************************************************************
    % the delay estimate is the final location of the delay that 
    % maximized crosscorrelation (center of window).
    % ****************************************************************
    delay_samples = x2;

    % ****************************************************************
    % Coefficient: Turn signal 1 into signal 2
    % ****************************************************************
    coeff = fd2Tau * fd1' ./ (fd1 * fd1');

    % ****************************************************************
    % chop roundoff error, if input signals are known to be 
    % real-valued.
    % ****************************************************************
    if rflag
        coeff = real(coeff);
    end
end

% ****************************************************************
% delay cyclic signal by phase shift
% ****************************************************************
function waveform = cs_delay(waveform, rate_Hz, delay_s)
    rflag = isreal(waveform);
    
    n = numel(waveform);
    cycLen_s = n / rate_Hz;
    nCyc = delay_s / cycLen_s();
    
    f = 0:(n - 1);
    f = f + floor(n / 2);
    f = mod(f, n);
    f = f - floor(n / 2);
    
    phase = -2 * pi * f * nCyc;
    rot = exp(1i*phase);
    
    waveform = ifft(fft(waveform) .* rot);
    if rflag
        waveform = real(waveform);
    end
end