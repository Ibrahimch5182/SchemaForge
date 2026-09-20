import { useEffect } from "react";
import { useRoute } from "./lib/route";
import { Landing } from "./pages/Landing";
import { Workspace } from "./pages/Workspace";

export default function App() {
  const [route, navigate] = useRoute();

  useEffect(() => {
    document.title = route === "workspace" ? "Workspace — SchemaForge" : "SchemaForge — Text-to-SQL on AWS EC2";
  }, [route]);

  return (
    <>
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      {route === "workspace" ? <Workspace navigate={navigate} /> : <Landing navigate={navigate} />}
    </>
  );
}
