import React, { useState, useEffect } from "react";
import VideoItem from "./VideoItem";
import BiasAlertDialog from "./BiasAlertDialog";

interface Video {
  id: number;
  videoUrl: string;
  image: string;
  name: string;
  verified: boolean;
  profile: string;
  likes: number;
  comments: number;
  bookmarks: number;
  shares: number;
  username: string;
  description: string;
  political_bias_category: string;
  religious_bias_category: string;
  overall_bias_category: string;
}

const videoData: Video[] = [
  {
    id: 1,
    videoUrl: "videos/v1.mp4",
    image: "/images/image1.jpg",
    name: "User1",
    verified: true,
    profile: "/images/profile1.jpg",
    likes: 12,
    comments: 5,
    bookmarks: 3,
    shares: 2,
    username: "dailymailuk",
    description: "Donald Trump appeared to interrupt British PM...",
    political_bias_category: "High Bias",
    religious_bias_category: "High Bias",
    overall_bias_category: "High Bias",
  },
  {
    id: 2,
    videoUrl: "videos/video2.mp4",
    image: "/images/image2.jpg",
    name: "User2",
    verified: false,
    profile: "/images/profile2.jpg",
    likes: 20,
    comments: 8,
    bookmarks: 5,
    shares: 3,
    username: "khafan67",
    description: "hehe #gun ...",
    political_bias_category: "Low Bias",
    religious_bias_category: "High Bias",
    overall_bias_category: "High Bias",
  },
  {
    id: 2,
    videoUrl: "videos/v3.mp4",
    image: "/images/image2.jpg",
    name: "User2",
    verified: false,
    profile: "/images/profile2.jpg",
    likes: 20,
    comments: 8,
    bookmarks: 5,
    shares: 3,
    username: "earthwidenews",
    description: "#fyp",
    political_bias_category: "High Bias",
    religious_bias_category: "Low Bias",
    overall_bias_category: "High Bias",
  },
  {
    id: 2,
    videoUrl: "videos/v4.mp4",
    image: "/images/image2.jpg",
    name: "User2",
    verified: false,
    profile: "/images/profile2.jpg",
    likes: 20,
    comments: 8,
    bookmarks: 5,
    shares: 3,
    username: "kingkhieu",
    description: "??The Crisis (Wait for it..)",
    political_bias_category: "High Bias",
    religious_bias_category: "Low Bias",
    overall_bias_category: "High Bias",
  },
  {
    id: 2,
    videoUrl: "videos/v5.mp4",
    image: "/images/image2.jpg",
    name: "User2",
    verified: false,
    profile: "/images/profile2.jpg",
    likes: 20,
    comments: 8,
    bookmarks: 5,
    shares: 3,
    username: "7newsaustalia",
    description: "Water in Sydney Harbour has turned bright red, ...",
    political_bias_category: "Low Bias",
    religious_bias_category: "High Bias",
    overall_bias_category: "Low Bias",
  },
];

const calculateBiasness = (videoData: Video[]) => {
  return videoData.reduce((count, video) => {
    return video.overall_bias_category === "High Bias" ? count + 1 : count;
  }, 0);
};

const VideoContent = () => {
  const [isAlertOpen, setIsAlertOpen] = useState(false);

  const biasness = calculateBiasness(videoData);

  console.log(biasness);

  useEffect(() => {
    if (biasness > 3) {
      const timer = setTimeout(() => {
        setIsAlertOpen(true);
      }, 2000);

      return () => clearTimeout(timer);
    }
  }, [biasness]);

  const handleNeutralizeFeed = () => {
    setIsAlertOpen(false);
  };

  return (
    <div className="h-[600px] w-[450px] scrollbar-hide snap-y snap-mandatory overflow-y-scroll">
      <BiasAlertDialog
        isOpen={isAlertOpen}
        onClose={() => setIsAlertOpen(false)}
        onNeutralize={handleNeutralizeFeed}
      />
      {videoData.map((video) => (
        <VideoItem
          key={video.id}
          src={video.videoUrl}
          userName={video.username}
          description={video.description}
          political_bias_category={video.political_bias_category} religious_bias_category={video.religious_bias_category} overall_bias_category={video.overall_bias_category}        />
      ))}
    </div>
  );
};

export default VideoContent;
