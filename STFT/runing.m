% *******************************************************
% delay-matching between two signals (complex/real-valued)
% M. Nentwig
% 
% Example use of "fitSignal"
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
% *******************************************************
function demo_fitSignal_120825();
    close all;
    n = 1024;    
    % n = 1024 * 256; disp('*** test: long signal enabled ***');
    opts = struct();
    
    % ****************************************************************
    % random signal
    % ****************************************************************
    fd = randn(1, n) + 1i * randn(1, n);
    
    % ****************************************************************
    % lowpass filter
    % ****************************************************************
    f = binFreq(n);
    fd(abs(f) > 0.045) = 0;
    s1 = real(ifft(fd)) * sqrt(n);
    
    % ****************************************************************
    % create delayed 2nd signal
    % ****************************************************************
    dTest_samples = 12.3456;
    cTest = 1.23456;
    % cTest = cTest + 1i; disp('*** test: complex coeff enabled ***'); 
    % cTest = -cTest; disp('*** test: negative coeff enabled ***'); 
    
    s2 = cTest * cs_delay(s1, 1, dTest_samples);
    %s2 = s2 + 0.5*randn(size(s2)); disp('*** test: noise enabled ***');
    %opts.forceIterativeAlgorithm = true; % better choice for noise
    
    % ****************************************************************
    % estimate delay
    % ****************************************************************
    [coeff, s2bb, delay_samples] = fitSignal_120825(s1, s2, opts);
    
    % ****************************************************************
    % correct it, and scale back for best fit
    % The above function call already calculates s2b (s2bb). Here,
    % calculate it again and show delay and scaling separately for the
    % plot.
    % ****************************************************************
    % Definition for the returned delay and coeff:
    % turn s2 into s1
    s2a = cs_delay(s2, 1, delay_samples);
    s2b = s2a * coeff;

    figure(); hold on;
    h = plot(real(s1), 'k'); set(h, 'lineWidth', 3);
    h = plot(real(s2), 'b'); set(h, 'lineWidth', 3);
    h = plot(real(s2a), 'r'); set(h, 'lineWidth', 1);
    h = plot(real(s2b), 'm'); set(h, 'lineWidth', 2);
    h = plot(real(s2bb), 'y'); set(h, 'lineWidth', 1);
    xlim([1, numel(s1)]);
    xlabel('samples');
    legend('s1', 's2', 's2 un-delayed', 's2 un-delayed and scaled', 'returned un-delayed/scaled s2');
    title('test signals');
    
    format long;
    disp('nominal delay of s2 relative to s1');
    dTest_samples
    disp('iterDelayEst() returned (negated):');
    -delay_samples
    disp('original scaling factor:');
    cTest
    disp('estimated scaling factor (inverted):');
    1 / coeff
end

% ****************************************************************
% frequency corresponding to FFT bin
% ****************************************************************
function f = binFreq(n)
    f = (mod(((0:n-1)+floor(n/2)), n)-floor(n/2))/n;
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