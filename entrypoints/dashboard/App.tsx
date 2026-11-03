import Sidebar from './components/Sidebar';
import WelcomeSection from './components/WelcomeSection';
import WritingComposer from './components/WritingComposer';

import heroBackground from '../../assets/hero-background.png';

function App() {
  return (
    <div className="flex h-dvh w-full overflow-hidden bg-[#f8f6ef]">

      <Sidebar />

      <main
        className="relative flex min-w-0 flex-1 flex-col overflow-y-auto"
        style={{
          backgroundImage: `url(${heroBackground})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
        }}
      >
        {/* Soft overlay to keep the content readable */}
        <div className="pointer-events-none fixed inset-y-0 right-0 z-0 bg-[#f8f6ef]/75"
          style={{ left: '224px' }}
        />

        <div className="relative z-10 flex min-h-full flex-1 flex-col">
          <WelcomeSection />

          <WritingComposer />

        </div>
      </main>

    </div>
  );
}

export default App;