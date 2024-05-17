import { ComponentFixture, TestBed } from '@angular/core/testing';
import { FormsModule, ReactiveFormsModule, FormGroupDirective } from '@angular/forms';

// primeng
import { DropdownModule } from 'primeng/dropdown';

// components and services
import { DemographicsSurveyComponent } from './demographics-survey.component';
import { MultiSelectModule } from 'primeng/multiselect';

describe('DemographicsSurveyComponent', () => {
  let component: DemographicsSurveyComponent;
  let fixture: ComponentFixture<DemographicsSurveyComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [
        DemographicsSurveyComponent,
      ],
      imports: [
        DropdownModule,
        MultiSelectModule,
        FormsModule,
        ReactiveFormsModule,
      ],
      providers: [
        FormGroupDirective
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DemographicsSurveyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
