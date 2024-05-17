import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HomepageComponent } from './homepage.component';
import { NavbarComponent } from '../navbar/navbar.component';
import { AppHeaderComponent } from '../app-header/app-header.component';
import { FooterComponent } from '../footer/footer.component';
import { MenubarModule } from 'primeng/menubar';
import { ActivatedRoute } from '@angular/router';

describe('HomepageComponent', () => {
  let component: HomepageComponent;
  let fixture: ComponentFixture<HomepageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [HomepageComponent, NavbarComponent, AppHeaderComponent, FooterComponent],
      imports: [MenubarModule],
      providers: [
        { provide: ActivatedRoute, useValue: {} } // Mock ActivatedRoute without any specific data
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(HomepageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
