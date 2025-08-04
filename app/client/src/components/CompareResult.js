import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import GradeBadge       from './GradeBadge';
import perfectImg       from '../assets/Perfect.png';
import greatImg         from '../assets/Great.png';
import goodImg          from '../assets/Good.png';
import badImg           from '../assets/Bad.png';
import DownloadIcon     from '../assets/Download.png';

// MM:SS 포맷
const formatTime = sec => {
  const m = String(Math.floor(sec/60)).padStart(2,'0');
  const s = String(Math.floor(sec % 60)).padStart(2,'0');
  return `${m}:${s}`;
};

// 0~1 → 4단계
const getGrade = score => {
  const pct = score * 100;
  if (pct >= 90) return { label: 'Perfect', icon: perfectImg };
  if (pct >= 80) return { label: 'Great',   icon: greatImg   };
  if (pct >= 70) return { label: 'Good',    icon: goodImg    };
  return { label: 'Bad',     icon: badImg     };
};

export default function CompareResult() {
  const location = useLocation();
  const navigate = useNavigate();
  const videoRef = useRef();

  const [fps, setFps]                   = useState(30);
  const [frameScores, setFrameScores]   = useState({});
  const [secScores, setSecScores]       = useState({});
  const [lowestInSec, setLowestInSec]   = useState({});
  const [feedback, setFeedback]         = useState({});
  const [chunkAvgs, setChunkAvgs]       = useState([]);
  const [currentGrade, setCurrentGrade] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [currentFrameFeedback, setCurrentFrameFeedback] = useState([]);
  const [history, setHistory]           = useState([]);
  const [tab, setTab]                   = useState('feedback');
  const [showPanel, setShowPanel]       = useState(true);

  const threshold = 0.8;

  // guard
  const state = location.state;
  useEffect(() => {
    if (!state) {
      navigate('/', { replace: true });
      return;
    }
    const { scoresJson, feedbackJson } = state;

    fetch(scoresJson)
      .then(r => r.json())
      .then(j => {
        console.log(j)
        setFps(j.fps);
        setFrameScores(j.frame_scores);
        setSecScores(j.second_scores);
        setLowestInSec(j.second_lowest_frames);
        console.log('fps', fps)
        console.log('frame_scores', frameScores)
        console.log('second_scores', secScores)
        console.log('second_lowest_frames', lowestInSec)

        const cs   = Math.floor(j.fps / 3) || 1;
        const vals = Object.values(j.frame_scores).map(Number);
        const avgs = [];
        for (let i = 0; i < vals.length; i += cs) {
          const slice = vals.slice(i, i + cs);
          avgs.push(slice.reduce((a,b)=>a+b,0)/slice.length);
        }
        setChunkAvgs(avgs);
      });

    fetch(feedbackJson)
      .then(r => r.json())
      .then(j => setFeedback(j));
  }, [state, navigate]);

  if (!state) return null;
  const { finalVideo } = state;

  const onTimeUpdate = () => {
    if (!videoRef.current) return;
    if (!showPanel) setShowPanel(true);

    const sec      = videoRef.current.currentTime;
    const frameIdx = Math.round(sec * fps);
    setCurrentFrame(frameIdx);
    setCurrentFrameFeedback(feedback[frameIdx] || []);

    const cs  = Math.floor(fps/3) || 1;
    const idx = Math.floor(frameIdx / cs);
    const avg = chunkAvgs[idx];
    if (avg != null) setCurrentGrade(getGrade(avg));

    const secIdx = Math.floor(sec);
    const secAvg = secScores[secIdx];
    if (secAvg !== undefined && secAvg < threshold) {
      const badFrame = lowestInSec[secIdx];
      const msgs     = feedback[badFrame] || [];
      if (msgs.length) {
        const key = `${secIdx}#${badFrame}`;
        if (!history.some(e => e.key === key)) {
          setHistory(h => [
            ...h,
            { key, time: formatTime(secIdx), frame: badFrame, msgs }
          ]);
        }
      }
    }
  };

  const seekTo = entry => {
    if (!videoRef.current) return;
    videoRef.current.pause();
    videoRef.current.currentTime = entry.frame / fps;
    setCurrentFrame(entry.frame);
    setCurrentFrameFeedback(entry.msgs);
  };

  const handleDownload = () => {
    const data = JSON.stringify({ frameScores, feedback }, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'scores_feedback.json';
    a.click();
    URL.revokeObjectURL(url);
  };

   return (
     <div>
       {/* Background circles */}

 
       {/* Header */}

 
       <div className="flex h-[80vh] mt-8">
         {/* 왼쪽: 시연용 최종 영상 */}
         <div className="w-1/2 p-4">
           <video
             ref={videoRef}
             src={finalVideo}
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
 
             {/* <div className="bg-white p-4 rounded shadow-sm h-1/3"> */}
             <div className="bg-white/20 backdrop-blur-md p-4 rounded-2xl shadow-lg h-1/3">
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
 
             {/* <div className="flex-1 overflow-y-auto bg-yellow-50 p-4 rounded shadow-sm"> */}
             <div className="flex-1 overflow-y-auto
                 scrollbar            /* 얇은 스크롤바 */
                scrollbar-thumb-violet-400 /* thumb 색상 */
              scrollbar-track-transparent /* 트랙은 투명 */
              scrollbar-thumb-rounded  /* 모서리 둥글게 */
              bg-pink-50/20 backdrop-blur-md p-4 rounded-2xl shadow-lg">
               <h3 className="font-semibold mb-2">📋 피드백 히스토리</h3>
 
               {tab === 'feedback'
                 ? (
                   history.length === 0
                     ? <p className="text-gray-500">이상 구간이 없습니다.</p>
                     : history.map(entry => (
                       <div
                         key={entry.key}
                         onClick={() => seekTo(entry)}
                         className="mb-4 p-2 cursor-pointer hover:bg-pink-200 rounded"
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
     </div>
   );
 }
 