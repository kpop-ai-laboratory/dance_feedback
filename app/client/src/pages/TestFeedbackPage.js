import React, { useState, useEffect } from 'react';
import { Card, CardContent , Button } from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import ReactPlayer from 'react-player';

const FeedbackPage = () => {
  const [currentFrameScore, setCurrentFrameScore] = useState(95);
  const [threeSecFeedback, setThreeSecFeedback] = useState([
    { time: '0~3s', comment: '동작 전환이 자연스럽습니다.' },
    { time: '3~6s', comment: '손 위치가 조금 낮습니다.' },
  ]);
  const [liveFeedback, setLiveFeedback] = useState('자세가 매우 정확합니다.');

  useEffect(() => {
    const feedbacks = [
      '자세가 매우 정확합니다.',
      '조금 더 팔을 펴보세요.',
      '타이밍이 완벽합니다!',
    ];

    const interval = setInterval(() => {
      setLiveFeedback(feedbacks[Math.floor(Math.random() * feedbacks.length)]);
    }, 300);

    return () => clearInterval(interval);
  }, []);

  const getScoreEffect = (score) => {
    if (score >= 90) return 'Perfect 🎉';
    if (score >= 80) return 'Great ✨';
    if (score >= 70) return 'Good 👍';
    return 'Bad ⚠️';
  };

  return (
    <div className="min-h-screen bg-gray-100 py-10 px-20 font-sans">
      <div className="grid grid-cols-2 gap-8">
        <Card className="shadow-lg rounded-2xl overflow-hidden">
          <ReactPlayer
            url="/video/dance.mp4"
            width="100%"
            height="auto"
            controls
          />
        </Card>

        <div className="space-y-6">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-4xl font-bold text-center"
          >
            {getScoreEffect(currentFrameScore)}
          </motion.div>

          <Card className="shadow-md bg-black text-white">
            <CardContent>
              <div className="font-semibold text-lg">구간별 피드백 (3초)</div>
              {threeSecFeedback.map((fb, index) => (
                <div key={index} className="my-2">
                  <span className="font-bold">{fb.time}:</span> {fb.comment}
                </div>
              ))}
            </CardContent>
          </Card>

          <AnimatePresence mode="wait">
            <motion.div
              key={liveFeedback}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className="text-center text-xl font-medium text-blue-600"
            >
              {liveFeedback}
            </motion.div>
          </AnimatePresence>

          {/* <Button className="w-full bg-blue-500 hover:bg-blue-700 text-white font-semibold rounded-lg shadow">
            전체 프레임별 피드백 다운로드
          </Button> */}
        </div>
      </div>
    </div>
  );
};

export default FeedbackPage;