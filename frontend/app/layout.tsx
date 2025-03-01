import "./globals.css";
import SideBar from "@/components/shared/SideBar";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <section className="flex justify-start items-start">
          <SideBar />
          <main className="w-full min-h-screen pt-4">{children}</main>
        </section>
      </body>
    </html>
  );
}
