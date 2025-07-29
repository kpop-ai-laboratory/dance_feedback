// src/components/DemoCompareResult.jsx
import React, { useState, useEffect, useRef } from 'react';
import GradeBadge       from './GradeBadge';
import perfectImg       from '../assets/Perfect.png';
import greatImg         from '../assets/Great.png';
import goodImg          from '../assets/Good.png';
import badImg           from '../assets/Bad.png';
import DownloadIcon     from '../assets/Download.png';
import { Music, ArrowLeft } from 'lucide-react';

const jobId = "31f3cb2568044a29a89f382a847271de";

const formatTime = sec => {
  const m = String(Math.floor(sec / 60)).padStart(2, '0');
  const s = String(Math.floor(sec % 60)).padStart(2, '0');
  return `${m}:${s}`;
};

const getGrade = score => {
  const pct = score * 100;
  if (pct >= 90) return { label: 'Perfect', icon: perfectImg };
  if (pct >= 80) return { label: 'Great', icon: greatImg };
  if (pct >= 70) return { label: 'Good', icon: goodImg };
  return { label: 'Bad', icon: badImg };
};

export default function DemoCompareResult() {
  const videoRef = useRef();

  const [fps, setFps] = useState(30);
  const [frameScores, setFrameScores] = useState({});
  const [secScores, setSecScores] = useState({});
  const [lowestInSec, setLowestInSec] = useState({});
  const [feedback, setFeedback] = useState({});
  const [chunkAvgs, setChunkAvgs] = useState([]);
  const [currentGrade, setCurrentGrade] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [currentFrameFeedback, setCurrentFrameFeedback] = useState([]);
  const [history, setHistory] = useState([]);
  const [tab, setTab] = useState('feedback');
  const [showPanel, setShowPanel] = useState(true);

  const threshold = 0.8;

  useEffect(() => {
    const scoresJson = `/data/${jobId}/dancer_kp/scores.json`;
    const feedbackJson = `/data/${jobId}/dancer_kp/feedback.json`;

    fetch(scoresJson)
      .then(r => r.json())
      .then(j => {
        setFps(j.fps);
        setFrameScores(j.frame_scores);
        setSecScores(j.second_scores);
        setLowestInSec(j.second_lowest_frames);

        const cs = Math.floor(j.fps / 3) || 1;
        const vals = Object.values(j.frame_scores).map(Number);
        const avgs = [];
        for (let i = 0; i < vals.length; i += cs) {
          const slice = vals.slice(i, i + cs);
          avgs.push(slice.reduce((a, b) => a + b, 0) / slice.length);
        }
        setChunkAvgs(avgs);
      });

    fetch(feedbackJson)
      .then(r => r.json())
      .then(j => setFeedback(j));
  }, []);

  const onTimeUpdate = () => {
    if (!videoRef.current) return;
    if (!showPanel) setShowPanel(true);

    const sec = videoRef.current.currentTime;
    const frameIdx = Math.round(sec * fps);
    setCurrentFrame(frameIdx);
    setCurrentFrameFeedback(feedback[frameIdx] || []);

    const cs = Math.floor(fps / 3) || 1;
    const avg = chunkAvgs[Math.floor(frameIdx / cs)];
    if (avg != null) {
      setCurrentGrade(getGrade(avg));
    }

    const secIdx = Math.floor(sec);
    const secAvg = secScores[secIdx];
    if (secAvg !== undefined && secAvg < threshold) {
      const badFrame = lowestInSec[secIdx];
      const msgs = feedback[badFrame] || [];
      if (msgs.length && !history.some(e => e.key === `${secIdx}#${badFrame}`)) {
        setHistory(h => [
          ...h,
          {
            key: `${secIdx}#${badFrame}`,
            time: formatTime(secIdx),
            frame: badFrame,
            msgs
          }
        ]);
      }
    }
  };

  const seekTo = entry => {
    if (!videoRef.current) return;
    videoRef.current.pause();
    const targetTime = entry.frame / fps;
    videoRef.current.currentTime = targetTime;
    setCurrentFrame(entry.frame);
    setCurrentFrameFeedback(entry.msgs);
  };

  const handleDownload = () => {
    const data = JSON.stringify({ frameScores, feedback }, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'scores_feedback.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div >

      {/* Header */}


      <div className="flex h-[80vh] mt-8">
        {/* 왼쪽: 시연용 최종 영상 */}
        <div className="w-1/2 p-4">
          <video
            ref={videoRef}
            src={`/data/${jobId}/final_feedback_with_audio.mp4`}
            controls
            onTimeUpdate={onTimeUpdate}
            className="w-full h-full object-contain rounded-lg border"
          />
        </div>

        {/* 오른쪽 패널 */}
        {showPanel && (
          <div className="w-1/2 flex flex-col space-y-4 p-4">
            <div className="flex justify-end">
              <button onClick={() => setShowPanel(false)} className="text-2xl font-bold">×</button>
            </div>

            <div className="flex justify-center">
              {currentGrade && (
                <GradeBadge
                  label={currentGrade.label}
                  color={currentGrade.label.toLowerCase()}
                />
              )}
            </div>

            <div className="bg-white p-4 rounded shadow-sm h-1/3">
              <h4 className="font-semibold mb-2">Frame {currentFrame} Feedback</h4>
              {currentFrameFeedback.length === 0
                ? <p className="text-gray-500">피드백이 없습니다.</p>
                : (
                  <ul className="list-disc list-inside text-sm ml-4">
                    {currentFrameFeedback.map((m, i) => <li key={i}>{m}</li>)}
                  </ul>
                )}
            </div>

            <div className="flex justify-center space-x-4">
              {['feedback', 'all'].map(x => (
                <button
                  key={x}
                  onClick={() => setTab(x)}
                  className={`px-4 py-1 rounded ${tab === x ? 'bg-red-300 text-white' : 'bg-gray-200 text-gray-700'}`}
                >
                  {x === 'feedback' ? 'FeedBack History' : 'All History'}
                </button>
              ))}
            </div>

            <div className="flex-1 overflow-y-auto bg-yellow-50 p-4 rounded shadow-sm">
              <h3 className="font-semibold mb-2">📋 피드백 히스토리</h3>

              {tab === 'feedback'
                ? (
                  history.length === 0
                    ? <p className="text-gray-500">이상 구간이 없습니다.</p>
                    : history.map(entry => (
                      <div
                        key={entry.key}
                        onClick={() => seekTo(entry)}
                        className="mb-4 p-2 cursor-pointer hover:bg-yellow-200 rounded"
                      >
                        <div className="flex items-center">
                          <span className="underline text-sm mr-2">{entry.time}</span>
                          <span className="text-xs text-gray-600 ml-auto">
                            frame #{entry.frame}{' '}
                            score {(frameScores[entry.frame] * 100).toFixed(1)}%
                          </span>
                        </div>
                        <ul className="list-disc list-inside text-sm ml-4 mt-1">
                          {entry.msgs.map((m, i) => <li key={i}>{m}</li>)}
                        </ul>
                      </div>
                    ))
                )
                : (
                  Object.entries(frameScores).map(([f, s]) => {
                    const frame = Number(f);
                    const msgs = feedback[frame] || ['피드백 없음'];
                    return (
                      <div
                        key={frame}
                        onClick={() => seekTo({ frame, msgs })}
                        className="mb-4 p-2 cursor-pointer hover:bg-yellow-200 rounded"
                      >
                        <div className="flex items-center">
                          <span className="text-sm">{`frame #${frame}`}</span>
                          <span className="text-xs text-gray-600 ml-auto">
                          </span>
                        </div>
                        <ul className="list-disc list-inside text-sm ml-4 mt-1">
                          {msgs.map((m, i) => <li key={i}>{m}</li>)}
                        </ul>
                      </div>
                    );
                  })
                )
              }
            </div>

            <div className="flex justify-center">
              <button
                onClick={handleDownload}
                className="inline-flex items-center px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                <img src={DownloadIcon} alt="" className="w-4 h-4 mr-2" />
                전체 다운로드
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      {/* <footer className="p-4 text-center text-white/70 drop-shadow-md">
        © 2025 K-pop Dance Feedback. All Rights Reserved.
      </footer> */}
    </div>
  );
}
