
import { useState } from 'react';

import Sidebar from './components/Sidebar';
import WelcomeSection from './components/WelcomeSection';
import WritingComposer from './components/WritingComposer';
import WritingData from './components/WritingData';

import heroBackground from '../../assets/hero-background.png';

type Section =
  | 'Dashboard'
  | 'My Writing Data'
  | 'My Writing Style'
  | 'Settings';

function App() {
  const [activeSection, setActiveSection] = useState<Section>('Dashboard');

  function renderContent() {
    if (activeSection === 'Dashboard') {
      return (
        <>
          <WelcomeSection />
          <WritingComposer />
        </>
      );
    }

    if (activeSection === 'My Writing Data') {
      return <WritingData />;
    }

    if (activeSection === 'My Writing Style') {
      return (
        <section className="px-12 py-12">
          <h1 className="font-serif text-4xl text-[#203c30]">
            My Writing Style
          </h1>
          <p className="mt-3 text-sm text-[#68766d]">
            Your personal writing style will be available here.
          </p>
        </section>
      );
    }

    return (
      <section className="px-12 py-12">
        <h1 className="font-serif text-4xl text-[#203c30]">
          Settings
        </h1>
        <p className="mt-3 text-sm text-[#68766d]">
          Your Echo preferences will be available here.
        </p>
      </section>
    );
  }

  return (
    <div className="flex h-dvh w-full overflow-hidden bg-[#f8f6ef]">
      <Sidebar
        activeSection={activeSection}
        onNavigate={setActiveSection}
      />

      <main
        className="relative flex min-w-0 flex-1 flex-col overflow-y-auto"
        style={{
          backgroundImage: `url(${heroBackground})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
        }}
      >
        <div
          className="pointer-events-none fixed inset-y-0 right-0 z-0 bg-[#f8f6ef]/75"
          style={{ left: '224px' }}
        />

        <div className="relative z-10 flex min-h-full flex-1 flex-col">
          {renderContent()}
        </div>
      </main>
    </div>
  );
}

export default App;
