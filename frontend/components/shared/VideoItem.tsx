import React, { useState, useRef } from "react";
import Image from "next/image";
import Profile from "./Profile";

type VideoItemProps = {
  src: string;
  userName: string;
  description: string;
  political_bias_category: string;
  religious_bias_category: string;
  overall_bias_category: string;
};

const VideoItem: React.FC<VideoItemProps> = ({
  src,
  userName,
  description,
  political_bias_category,
  religious_bias_category,
  overall_bias_category,
}) => {
  const [isPlaying, setIsPlaying] = useState(true);
  const [showPlayButton, setShowPlayButton] = useState(false);
  const [showPauseButton, setShowPauseButton] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
        setShowPauseButton(true);
        setTimeout(() => setShowPauseButton(false), 500);
      } else {
        videoRef.current.play();
        setShowPlayButton(true);
        setTimeout(() => setShowPlayButton(false), 500);
      }
      setIsPlaying(!isPlaying);
    }
  };

  const renderBiasImage = (x: string) => {
    if (x === "Low Bias") {
      return (
        <Image
          src="/images/green.svg"
          alt="Low Bias"
          width={20}
          height={20}
        />
      );
    }
    return (
      <Image
        src="/images/red.png"
        alt="High Bias"
        width={20}
        height={20}
      />
    );
  };

  return (
    <section className="w-[500px] h-[600px] relative snap-start">
      <video
        ref={videoRef}
        src={src}
        autoPlay
        loop
        muted
        playsInline
        className="w-[500px] h-[600px] cursor-pointer"
        onClick={togglePlayPause}
      />
      {showPlayButton && (
        <div className="absolute inset-0 flex items-center justify-center">
          <Image
            src="/icons/play-button.svg"
            alt="Play Button"
            width={50}
            height={50}
          />
        </div>
      )}

      {showPauseButton && (
        <div className="absolute inset-0 flex items-center justify-center">
          <Image
            src="/icons/pause-button-icon.svg"
            alt="Pause Button"
            width={50}
            height={50}
          />
        </div>
      )}

      {/* Render political bias images inline */}
      <div className="absolute top-5 right-20 pr-2 flex flex-col">
        <div className="flex items-center justify-start space-x-2 px-2 gap-1">
        {renderBiasImage(political_bias_category)}
        {renderBiasImage(religious_bias_category)}
        {renderBiasImage(overall_bias_category)}
        </div>
        <div className="flex flex-row gap-1">
        <div className="text-[8px] text-white">
        political
        </div>
        <div className="text-[8px] text-white">
        religion
        </div>
        <div className="text-[8px] text-white">
        overall
        </div>
        </div>
      </div>

      <Profile userName={userName} description={description} />
    </section>
  );
};

export default VideoItem;
