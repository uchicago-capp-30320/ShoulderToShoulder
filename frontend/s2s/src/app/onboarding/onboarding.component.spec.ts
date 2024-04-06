import { ComponentFixture, TestBed } from '@angular/core/testing';

import { OnboardingComponent } from './onboarding.component';
import { FooterComponent } from '../footer/footer.component';
import { DemographicsSurveyComponent } from '../demographics-survey/demographics-survey.component';
import { ButtonModule } from 'primeng/button';

describe('OnboardingComponent', () => {
  let component: OnboardingComponent;
  let fixture: ComponentFixture<OnboardingComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [OnboardingComponent, FooterComponent, DemographicsSurveyComponent],
      imports: [ButtonModule]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(OnboardingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
