import React from "react";

type ProfileProps = {
  userName: string;
  description: string;
};
const Profile: React.FC<ProfileProps> = (props: ProfileProps) => {
  return (
    <div className=" pl-3 pb-5 w-[339px] h-[50px] absolute bottom-0 left-20 right-0 bg-gradient-to-t from-black to-transparent text-white">
      <div className="text-white font-bold">
        {props.userName}
      </div>
      <div className="text-sm ">{props.description}</div>
    </div>
  );
};

export default Profile;
